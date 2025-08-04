#!/usr/bin/env python3
"""
Export COLMAP reconstruction or StreamVGGT predictions to PLY format.

This script can export point clouds from:
1. StreamVGGT model predictions (from predictions.npz files)
2. COLMAP reconstruction data

Usage:
    python tools/export_colmap_ply.py <target_dir> [--out_dir <output_dir>]
"""

import argparse
import os
import sys
import numpy as np
import time
import gc
import glob
from pathlib import Path

# Add src to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import torch and other ML dependencies
try:
    import torch
    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    import trimesh
    from visual_util import predictions_to_glb
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have installed all requirements and the PYTHONPATH is set correctly.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading StreamVGGT model...")
local_ckpt_path = "ckpt/checkpoints.pth"
if os.path.exists(local_ckpt_path):
    print(f"Loading local checkpoint from {local_ckpt_path}")
    model = StreamVGGT()
    ckpt = torch.load(local_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
else:
    print("Local checkpoint not found, downloading from Hugging Face...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="lch01/StreamVGGT",
        filename="checkpoints.pth",
        revision="main",
        force_download=True
    )
    model = StreamVGGT()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    del ckpt


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check with fallback
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for CUDA compatibility issues
    if device == "cuda":
        try:
            # Test a simple CUDA operation
            test_tensor = torch.tensor([1.0]).cuda()
            _ = test_tensor + 1
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        except RuntimeError as e:
            if "no kernel image is available" in str(e) or "CUDA error" in str(e):
                print(f"⚠️ CUDA compatibility issue detected: {e}")
                print("🔄 Falling back to CPU processing...")
                device = "cpu"
            else:
                raise e

    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    predictions = {}    
    predictions["images"] = images  # (S, 3, H, W)
    print(f"Images shape: {images.shape}")

    frames = []
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0) 
        frame = {
            "img": image
        }
        frames.append(frame)

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast('cuda', dtype=dtype):
                output = model.inference(frames)
        else:
            # CPU inference without autocast
            output = model.inference(frames)

    all_pts3d = []
    all_conf = []
    all_depth = []
    all_depth_conf = []
    all_camera_pose = []
    
    for res in output.ress:
        all_pts3d.append(res['pts3d_in_other_view'].squeeze(0))
        all_conf.append(res['conf'].squeeze(0))
        all_depth.append(res['depth'].squeeze(0))
        all_depth_conf.append(res['depth_conf'].squeeze(0))
        all_camera_pose.append(res['camera_pose'].squeeze(0))

    predictions["world_points"] = torch.stack(all_pts3d, dim=0)  # (S, H, W, 3)
    predictions["world_points_conf"] = torch.stack(all_conf, dim=0)  # (S, H, W)
    predictions["depth"] = torch.stack(all_depth, dim=0)  # (S, H, W, 1)
    predictions["depth_conf"] = torch.stack(all_depth_conf, dim=0)  # (S, H, W)
    predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0)  # (S, 9)

    print("World points shape:", predictions["world_points"].shape)
    print("World points confidence shape:", predictions["world_points_conf"].shape)
    print("Depth map shape:", predictions["depth"].shape)
    print("Depth confidence shape:", predictions["depth_conf"].shape)
    print("Pose encoding shape:", predictions["pose_enc"].shape)
    print(f"Images shape: {images.shape}")
    
    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic.squeeze(0)  # (S, 3, 4)
    predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None  # (S, 3, 3) or None
    print("Extrinsic shape:", predictions["extrinsic"].shape)
    print("Intrinsic shape:", predictions["intrinsic"].shape)

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy()#.squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    #depth_map = predictions["depth"]  # (S, H, W, 1)
    #world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    #predictions["world_points_from_depth"] = world_points
    predictions["world_points_from_depth"] = predictions["world_points"]

    # Clean up
    torch.cuda.empty_cache()
    return predictions


def export_predictions_to_ply(target_dir, out_dir=None, conf_thres=50.0):
    """Export StreamVGGT predictions to PLY format."""

    predictions_path = os.path.join(target_dir, "predictions.npz")

    if not os.path.exists(predictions_path):
        print(f"Error: No predictions.npz found at {predictions_path}")
        return False
    
    print(f"Loading predictions from {predictions_path}")
    
    # Load predictions
    key_list = [
        "pose_enc", "depth", "depth_conf", "world_points", "world_points_conf",
        "images", "extrinsic", "intrinsic", "world_points_from_depth"
    ]
    
    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list if key in loaded}
    
    # Convert to GLB first using existing functionality
    print("Converting predictions to 3D scene...")
    glb_scene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames="all",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=False,  # Don't include cameras in PLY export
        mask_sky=False,
        target_dir=target_dir,
        prediction_mode="Predicted Pointmap"
    )
    
    # Extract point cloud from scene
    point_clouds = []
    for geometry_name, geometry in glb_scene.geometry.items():
        if hasattr(geometry, 'vertices') and len(geometry.vertices) > 0:
            # Create a point cloud from the mesh vertices
            vertices = geometry.vertices
            colors = None
            
            if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'vertex_colors'):
                colors = geometry.visual.vertex_colors[:, :3]  # RGB only
            elif hasattr(geometry, 'visual') and hasattr(geometry.visual, 'face_colors'):
                # If we have face colors, use them for vertices (approximate)
                colors = geometry.visual.face_colors[:len(vertices), :3]
            
            point_cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
            point_clouds.append(point_cloud)
    
    if not point_clouds:
        print("Warning: No point clouds found in the scene")
        return False
    
    # Combine all point clouds
    if len(point_clouds) == 1:
        combined_cloud = point_clouds[0]
    else:
        all_vertices = np.vstack([pc.vertices for pc in point_clouds])
        all_colors = None
        if all(pc.colors is not None for pc in point_clouds):
            all_colors = np.vstack([pc.colors for pc in point_clouds])
        combined_cloud = trimesh.PointCloud(vertices=all_vertices, colors=all_colors)
    
    # Export to PLY
    output_file = os.path.join(out_dir, "reconstruction.ply")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Exporting point cloud to {output_file}")
    combined_cloud.export(output_file)
    
    print(f"Successfully exported {len(combined_cloud.vertices)} points to {output_file}")
    return True

def compute_predictions(target_dir):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    return predictions

def main():
    parser = argparse.ArgumentParser(description="Export StreamVGGT predictions to PLY format")
    parser.add_argument("--conf_thres", type=float, default=50.0, 
                       help="Confidence threshold for filtering points (default: 50.0)")
    
    args = parser.parse_args()
    
    target_dir = "/working"

    if not os.path.exists(target_dir):
        print(f"Error: Target directory {target_dir} does not exist")
        sys.exit(1)

    out_dir = os.path.join(target_dir, "output")

    predictions = compute_predictions(target_dir)
    # Save predictions
    predictions_file = os.path.join(out_dir, "predictions.npz")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(predictions_file, **predictions)

    if os.path.exists(predictions_file):
        print("Found StreamVGGT predictions, exporting to PLY...")
        success = export_predictions_to_ply(out_dir, out_dir, args.conf_thres)
    else:
        print(f"Error: No supported data found in {target_dir}")
        print("Supported formats:")
        print("  - StreamVGGT predictions (predictions.npz)")
        success = False

    if success:
        print("Export completed successfully!")
    else:
        print("Export failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 