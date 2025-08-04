#!/usr/bin/env python3

import numpy as np
import struct
import os
import sys

def write_binary_cameras(cameras_dict, output_path):
    """Write cameras in COLMAP binary format."""
    with open(output_path, 'wb') as f:
        # Write number of cameras
        f.write(struct.pack('<Q', len(cameras_dict)))
        
        for camera_id, (model, width, height, params) in cameras_dict.items():
            # Write camera_id (uint32)
            f.write(struct.pack('<I', camera_id))
            
            # Write model_id (int32) - SIMPLE_PINHOLE = 0
            model_id = 0 if model == "SIMPLE_PINHOLE" else -1
            f.write(struct.pack('<i', model_id))
            
            # Write width, height (uint64)
            f.write(struct.pack('<Q', width))
            f.write(struct.pack('<Q', height))
            
            # Write parameters (double array)
            for param in params:
                f.write(struct.pack('<d', param))

def write_binary_images(images_dict, image_points_dict, output_path):
    """Write images in COLMAP binary format."""
    with open(output_path, 'wb') as f:
        # Write number of images
        f.write(struct.pack('<Q', len(images_dict)))
        
        for image_id, (qw, qx, qy, qz, tx, ty, tz, camera_id, name) in images_dict.items():
            # Write image_id (uint32)
            f.write(struct.pack('<I', image_id))
            
            # Write quaternion (4 x double)
            f.write(struct.pack('<d', qw))
            f.write(struct.pack('<d', qx))
            f.write(struct.pack('<d', qy))
            f.write(struct.pack('<d', qz))
            
            # Write translation (3 x double)
            f.write(struct.pack('<d', tx))
            f.write(struct.pack('<d', ty))
            f.write(struct.pack('<d', tz))
            
            # Write camera_id (uint32)
            f.write(struct.pack('<I', camera_id))
            
            # Write image name (null-terminated string)
            name_bytes = name.encode('utf-8') + b'\x00'
            f.write(name_bytes)
            
            # Write 2D points
            points_2d = image_points_dict.get(image_id, [])
            f.write(struct.pack('<Q', len(points_2d)))
            
            for x, y, point3d_id in points_2d:
                f.write(struct.pack('<d', x))      # x coordinate
                f.write(struct.pack('<d', y))      # y coordinate
                f.write(struct.pack('<Q', point3d_id))  # point3D_id

def write_binary_points3d(points3d_dict, output_path):
    """Write points3D in COLMAP binary format."""
    with open(output_path, 'wb') as f:
        # Write number of points
        f.write(struct.pack('<Q', len(points3d_dict)))
        
        for point3d_id, (x, y, z, r, g, b, error, track_data) in points3d_dict.items():
            # Write point3D_id (uint64)
            f.write(struct.pack('<Q', point3d_id))
            
            # Write XYZ coordinates (3 x double)
            f.write(struct.pack('<d', x))
            f.write(struct.pack('<d', y))
            f.write(struct.pack('<d', z))
            
            # Write RGB color (3 x uint8)
            f.write(struct.pack('<B', r))
            f.write(struct.pack('<B', g))
            f.write(struct.pack('<B', b))
            
            # Write error (double)
            f.write(struct.pack('<d', error))
            
            # Write track (image_id, point2D_idx pairs)
            track_length = len(track_data) // 2
            f.write(struct.pack('<Q', track_length))
            
            for i in range(track_length):
                image_id = int(track_data[i*2])
                point2d_idx = int(track_data[i*2 + 1])
                f.write(struct.pack('<I', image_id))
                f.write(struct.pack('<I', point2d_idx))

def convert_text_to_binary(text_dir, binary_dir):
    """Convert COLMAP text format to binary format."""
    
    print(f"Converting {text_dir} -> {binary_dir}")
    os.makedirs(binary_dir, exist_ok=True)
    
    # Read cameras.txt
    print("📷 Reading cameras.txt...")
    cameras = {}
    with open(os.path.join(text_dir, "cameras.txt"), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = (model, width, height, params)
    
    print(f"✅ Found {len(cameras)} cameras")
    
    # Read images.txt
    print("📸 Reading images.txt...")
    images = {}
    image_points = {}
    with open(os.path.join(text_dir, "images.txt"), 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        # Parse image line
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]
        
        images[image_id] = (qw, qx, qy, qz, tx, ty, tz, camera_id, name)
        
        # Parse 2D points line
        i += 1
        if i < len(lines):
            points_line = lines[i].strip()
            if points_line:
                point_data = points_line.split()
                num_points = len(point_data) // 3
                points_2d = []
                for j in range(num_points):
                    x = float(point_data[j*3])
                    y = float(point_data[j*3 + 1])
                    point3d_id = int(point_data[j*3 + 2])
                    points_2d.append((x, y, point3d_id))
                image_points[image_id] = points_2d
        i += 1
    
    print(f"✅ Found {len(images)} images")
    
    # Read points3D.txt
    print("🎯 Reading points3D.txt...")
    points3d = {}
    with open(os.path.join(text_dir, "points3D.txt"), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            point3d_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            error = float(parts[7])
            track_data = parts[8:]
            
            points3d[point3d_id] = (x, y, z, r, g, b, error, track_data)
    
    print(f"✅ Found {len(points3d)} 3D points")
    
    # Write binary files
    print("\n💾 Writing binary files...")
    write_binary_cameras(cameras, os.path.join(binary_dir, "cameras.bin"))
    print("✅ Wrote cameras.bin")
    
    write_binary_images(images, image_points, os.path.join(binary_dir, "images.bin"))
    print("✅ Wrote images.bin")
    
    write_binary_points3d(points3d, os.path.join(binary_dir, "points3D.bin"))
    print("✅ Wrote points3D.bin")
    
    print(f"\n🎉 Conversion complete! Binary files saved to: {binary_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_binary.py <text_dir> <binary_dir>")
        sys.exit(1)
    
    text_dir = sys.argv[1]
    binary_dir = sys.argv[2]
    convert_text_to_binary(text_dir, binary_dir) 