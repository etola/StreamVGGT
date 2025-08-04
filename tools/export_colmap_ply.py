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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'datasets_preprocess'))

# Import torch and other ML dependencies
try:
    import torch
    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    import trimesh
    from visual_util import predictions_to_glb
    # Import COLMAP utilities
    from read_write_model import Camera, Image, Point3D, write_cameras_text, write_images_text, write_points3D_text, rotmat2qvec
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


def export_colmap_sparse(predictions, out_dir, image_names):
    """Export predictions to COLMAP sparse format with proper multi-view correspondences."""
    
    print("Exporting to COLMAP sparse format...")
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Get data from predictions
    extrinsics = predictions["extrinsic"]  # (S, 3, 4)
    intrinsics = predictions["intrinsic"]  # (S, 3, 3)
    world_points = predictions["world_points"]  # (S, H, W, 3)
    world_points_conf = predictions["world_points_conf"]  # (S, H, W)
    images_tensor = predictions["images"]  # (S, 3, H, W)
    
    num_images = extrinsics.shape[0]
    height, width = world_points.shape[1:3]
    
    print(f"Processing {num_images} images with resolution {width}x{height}")
    
    # 1. Create cameras.txt
    cameras = {}
    for i in range(num_images):
        # Use SIMPLE_PINHOLE model (focal_length, cx, cy)
        if intrinsics is not None:
            fx = intrinsics[i, 0, 0]
            fy = intrinsics[i, 1, 1]
            cx = intrinsics[i, 0, 2]
            cy = intrinsics[i, 1, 2]
            focal = (fx + fy) / 2  # Average focal length for SIMPLE_PINHOLE
            params = [focal, cx, cy]
        else:
            # Default intrinsics if not available
            focal = min(width, height) * 0.7  # Rough estimate
            params = [focal, width/2, height/2]
        
        camera = Camera(
            id=i,
            model="SIMPLE_PINHOLE",
            width=width,
            height=height,
            params=params
        )
        cameras[i] = camera
    
    # 2. Sample 3D points from the first image and find correspondences in other images
    print("Finding multi-view correspondences...")
    
    # Sample confident 3D points from the first image
    ref_image_idx = 0
    conf_threshold = np.percentile(world_points_conf[ref_image_idx].flatten(), 85)  # Top 15% confident points
    conf_mask = world_points_conf[ref_image_idx] > conf_threshold
    
    # Get 2D coordinates and 3D points from reference image
    y_coords, x_coords = np.where(conf_mask)
    ref_points_3d = world_points[ref_image_idx][conf_mask]
    
    # Sample to avoid too many points (limit to ~500 points total)
    if len(ref_points_3d) > 500:
        indices = np.random.choice(len(ref_points_3d), 500, replace=False)
        ref_points_3d = ref_points_3d[indices]
        y_coords = y_coords[indices]
        x_coords = x_coords[indices]
    
    print(f"Using {len(ref_points_3d)} reference 3D points from image {ref_image_idx}")
    
    # Get colors from reference image
    ref_img_tensor = images_tensor[ref_image_idx]  # (3, H, W)
    ref_img = ref_img_tensor.transpose(1, 2, 0)  # (H, W, 3)
    ref_img = np.clip(ref_img, 0, 1)
    ref_img_rgb = (ref_img * 255).astype(np.uint8)
    
    # For each reference 3D point, find correspondences in other images
    valid_points_3d = []
    valid_point_colors = []
    valid_tracks = []  # List of (image_id, x, y) tuples for each 3D point
    
    for point_idx, (point_3d, ref_y, ref_x) in enumerate(zip(ref_points_3d, y_coords, x_coords)):
        track = [(ref_image_idx, ref_x, ref_y)]  # Start with reference observation
        
        # Project this 3D point to all other camera views
        for target_img_idx in range(1, num_images):
            # Get camera parameters for target image
            R_target = extrinsics[target_img_idx, :3, :3]
            t_target = extrinsics[target_img_idx, :3, 3]
            K_target = intrinsics[target_img_idx] if intrinsics is not None else cameras[target_img_idx]
            
            if not isinstance(K_target, np.ndarray):
                K_target = np.array([[cameras[target_img_idx].params[0], 0, cameras[target_img_idx].params[1]], 
                                   [0, cameras[target_img_idx].params[0], cameras[target_img_idx].params[2]], 
                                   [0, 0, 1]])
            
            # Transform 3D point to target camera coordinates
            point_3d_cam = R_target @ point_3d + t_target
            
            # Check if point is in front of camera
            if point_3d_cam[2] <= 0:
                continue
                
            # Project to image coordinates
            x_proj = K_target[0, 0] * point_3d_cam[0] / point_3d_cam[2] + K_target[0, 2]
            y_proj = K_target[1, 1] * point_3d_cam[1] / point_3d_cam[2] + K_target[1, 2]
            
            # Check if projection is within image bounds with some margin
            margin = 10
            if margin <= x_proj < width - margin and margin <= y_proj < height - margin:
                # Check if there's a valid 3D point at this projected location
                proj_x_int, proj_y_int = int(round(x_proj)), int(round(y_proj))
                target_conf = world_points_conf[target_img_idx][proj_y_int, proj_x_int]
                
                # Only add to track if confidence is reasonable at projected location
                if target_conf > conf_threshold * 0.5:  # Lower threshold for correspondences
                    track.append((target_img_idx, x_proj, y_proj))
        
        # Only keep 3D points that are visible in at least 2 images
        if len(track) >= 2:
            valid_points_3d.append(point_3d)
            
            # Get color from reference image
            rgb = ref_img_rgb[ref_y, ref_x]
            if len(rgb.shape) == 0:
                rgb = np.array([rgb, rgb, rgb])
            elif len(rgb) != 3:
                rgb = np.array([128, 128, 128])
            valid_point_colors.append(rgb)
            
            valid_tracks.append(track)
    
    print(f"Found {len(valid_points_3d)} 3D points with multi-view correspondences")
    
    if len(valid_points_3d) == 0:
        print("⚠️ No multi-view correspondences found! Using single-view points instead...")
        # Fallback: use points from first image only
        for point_idx, (point_3d, ref_y, ref_x) in enumerate(zip(ref_points_3d[:100], y_coords[:100], x_coords[:100])):
            valid_points_3d.append(point_3d)
            rgb = ref_img_rgb[ref_y, ref_x]
            if len(rgb.shape) == 0:
                rgb = np.array([rgb, rgb, rgb])
            elif len(rgb) != 3:
                rgb = np.array([128, 128, 128])
            valid_point_colors.append(rgb)
            valid_tracks.append([(ref_image_idx, ref_x, ref_y)])
    
    # 3. Create images.txt with proper 2D point correspondences
    images = {}
    
    for i in range(num_images):
        # Convert extrinsic matrix to COLMAP format (world-to-camera)
        R = extrinsics[i, :3, :3]  # Rotation matrix
        t = extrinsics[i, :3, 3]   # Translation vector
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        qvec = rotmat2qvec(R)
        
        # Get image name
        if i < len(image_names):
            image_name = os.path.basename(image_names[i])
        else:
            image_name = f"image_{i:04d}.jpg"
        
        # Collect 2D points for this image
        image_2d_points = []
        image_3d_point_ids = []
        
        for point_3d_id, track in enumerate(valid_tracks):
            for track_entry in track:
                if track_entry[0] == i:  # This 3D point is visible in image i
                    image_2d_points.append([track_entry[1], track_entry[2]])
                    image_3d_point_ids.append(point_3d_id)
                    break  # Only one observation per 3D point per image
        
        # Convert to numpy arrays
        if len(image_2d_points) > 0:
            xys = np.array(image_2d_points)
            point3D_ids = np.array(image_3d_point_ids, dtype=int)
        else:
            xys = np.zeros((0, 2))
            point3D_ids = np.zeros(0, dtype=int)
        
        image = Image(
            id=i,
            qvec=qvec,
            tvec=t,
            camera_id=i,  # Each image has its own camera
            name=image_name,
            xys=xys,
            point3D_ids=point3D_ids
        )
        images[i] = image
    
    # 4. Create points3D.txt with proper tracks
    points3D = {}
    
    for point_3d_id, (point_3d, rgb, track) in enumerate(zip(valid_points_3d, valid_point_colors, valid_tracks)):
        # Create track lists
        track_image_ids = []
        track_point2D_idxs = []
        
        for track_entry in track:
            image_id = track_entry[0]
            # Find the index of this 3D point in the image's 2D points
            image_obj = images[image_id]
            point_2d_indices = np.where(image_obj.point3D_ids == point_3d_id)[0]
            
            if len(point_2d_indices) > 0:
                track_image_ids.append(image_id)
                track_point2D_idxs.append(point_2d_indices[0])
        
        if len(track_image_ids) > 0:  # Only add points with valid tracks
            point3d = Point3D(
                id=point_3d_id,
                xyz=point_3d,
                rgb=rgb,
                error=0.1,  # Dummy error
                image_ids=np.array(track_image_ids),
                point2D_idxs=np.array(track_point2D_idxs)
            )
            points3D[point_3d_id] = point3d
    
    # Write COLMAP files
    print(f"Writing cameras.txt ({len(cameras)} cameras)...")
    write_cameras_text(cameras, os.path.join(sparse_dir, "cameras.txt"))
    
    print(f"Writing images.txt ({len(images)} images)...")
    write_images_text(images, os.path.join(sparse_dir, "images.txt"))
    
    print(f"Writing points3D.txt ({len(points3D)} points)...")
    write_points3D_text(points3D, os.path.join(sparse_dir, "points3D.txt"))
    
    print(f"✅ COLMAP sparse reconstruction saved to: {sparse_dir}")
    print(f"📊 Summary: {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")
    
    # Check for multi-view observations
    total_observations = sum(len(img.point3D_ids) for img in images.values())
    avg_track_length = total_observations / len(points3D) if len(points3D) > 0 else 0
    print(f"📈 Total 2D-3D observations: {total_observations}")
    print(f"📏 Average track length: {avg_track_length:.1f} views per 3D point")
    
    return sparse_dir


def compute_predictions(target_dir):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        print("No valid target directory found.")
        return None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    print("Running run_model...")
    try:
        with torch.no_grad():
            predictions = run_model(target_dir, model)
        
        # Get image names for COLMAP export
        image_names = glob.glob(os.path.join(target_dir, "images", "*"))
        image_names = sorted(image_names)

        return predictions, image_names
    except Exception as e:
        print(f"Error during model inference: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Export StreamVGGT predictions to PLY format")
    parser.add_argument("--conf_thres", type=float, default=50.0, 
                       help="Confidence threshold for filtering points (default: 50.0)")
    parser.add_argument("--export_colmap", action="store_true",
                       help="Export COLMAP sparse reconstruction format")
    
    args = parser.parse_args()
    
    target_dir = "/working"

    if not os.path.exists(target_dir):
        print(f"Error: Target directory {target_dir} does not exist")
        sys.exit(1)

    out_dir = os.path.join(target_dir, "output")

    predictions, image_names = compute_predictions(target_dir)
    
    if predictions is None:
        print("❌ Failed to generate predictions!")
        sys.exit(1)
    
    # Save predictions
    predictions_file = os.path.join(out_dir, "predictions.npz")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(predictions_file, **predictions)

    success = False
    
    if os.path.exists(predictions_file):
        print("Found StreamVGGT predictions, exporting...")
        
        # Export PLY format
        print("📦 Exporting to PLY format...")
        ply_success = export_predictions_to_ply(out_dir, out_dir, args.conf_thres)
        
        # Export COLMAP sparse format (always export for complete reconstruction)
        print("🏗️ Exporting to COLMAP sparse format...")
        try:
            sparse_dir = export_colmap_sparse(predictions, out_dir, image_names)
            colmap_success = True
            print(f"✅ COLMAP sparse reconstruction exported to: {sparse_dir}")
        except Exception as e:
            print(f"❌ COLMAP export failed: {e}")
            import traceback
            traceback.print_exc()
            colmap_success = False
        
        success = ply_success and colmap_success
        
    else:
        print(f"Error: No supported data found in {target_dir}")
        print("Supported formats:")
        print("  - StreamVGGT predictions (predictions.npz)")

    if success:
        print("\n🎉 Export completed successfully!")
        print(f"📁 Outputs saved to: {out_dir}")
        print("   📦 reconstruction.ply - Point cloud")
        print("   🏗️ sparse/0/ - COLMAP reconstruction")
        print("      - cameras.txt - Camera intrinsics")
        print("      - images.txt - Camera poses") 
        print("      - points3D.txt - 3D points")
    else:
        print("❌ Export failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 