#!/usr/bin/env python3

import numpy as np
import os
import sys

def validate_colmap_reconstruction(sparse_dir):
    """Validate COLMAP reconstruction files for common issues."""
    
    print(f"🔍 Validating COLMAP reconstruction in: {sparse_dir}")
    
    cameras_file = os.path.join(sparse_dir, "cameras.txt")
    images_file = os.path.join(sparse_dir, "images.txt")
    points3d_file = os.path.join(sparse_dir, "points3D.txt")
    
    issues = []
    
    # Check if files exist
    for filename, filepath in [("cameras.txt", cameras_file), ("images.txt", images_file), ("points3D.txt", points3d_file)]:
        if not os.path.exists(filepath):
            issues.append(f"❌ Missing file: {filename}")
            continue
        print(f"✅ Found: {filename}")
    
    if issues:
        return issues
    
    # Validate cameras.txt
    print("\n📷 Validating cameras.txt...")
    try:
        cameras = {}
        with open(cameras_file, 'r') as f:
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
                
                # Check for valid parameters
                if model == "SIMPLE_PINHOLE" and len(params) != 3:
                    issues.append(f"❌ Camera {camera_id}: SIMPLE_PINHOLE should have 3 parameters, got {len(params)}")
                
                if any(not np.isfinite(p) for p in params):
                    issues.append(f"❌ Camera {camera_id}: Non-finite parameters: {params}")
                
                if width <= 0 or height <= 0:
                    issues.append(f"❌ Camera {camera_id}: Invalid dimensions {width}x{height}")
                
                cameras[camera_id] = (model, width, height, params)
        
        print(f"✅ Found {len(cameras)} cameras")
        
    except Exception as e:
        issues.append(f"❌ Error reading cameras.txt: {e}")
    
    # Validate images.txt
    print("\n📸 Validating images.txt...")
    try:
        images = {}
        image_points = {}
        with open(images_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
            
            # Parse image line
            parts = line.split()
            if len(parts) < 10:
                issues.append(f"❌ Image line {i}: Insufficient data: {line}")
                i += 1
                continue
                
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]
            
            # Validate quaternion
            quat_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if abs(quat_norm - 1.0) > 1e-3:
                issues.append(f"❌ Image {image_id}: Quaternion not normalized: norm={quat_norm}")
            
            # Check for finite values
            if not all(np.isfinite([qw, qx, qy, qz, tx, ty, tz])):
                issues.append(f"❌ Image {image_id}: Non-finite pose values")
            
            # Check camera_id exists
            if camera_id not in cameras:
                issues.append(f"❌ Image {image_id}: References non-existent camera {camera_id}")
            
            images[image_id] = (qw, qx, qy, qz, tx, ty, tz, camera_id, name)
            
            # Parse 2D points line
            i += 1
            if i < len(lines):
                points_line = lines[i].strip()
                if points_line:
                    point_data = points_line.split()
                    if len(point_data) % 3 != 0:
                        issues.append(f"❌ Image {image_id}: 2D points data not divisible by 3: {len(point_data)}")
                    else:
                        num_points = len(point_data) // 3
                        points_2d = []
                        for j in range(num_points):
                            x = float(point_data[j*3])
                            y = float(point_data[j*3 + 1])
                            point3d_id = int(point_data[j*3 + 2])
                            
                            if not np.isfinite(x) or not np.isfinite(y):
                                issues.append(f"❌ Image {image_id}, point {j}: Non-finite 2D coordinates ({x}, {y})")
                            
                            points_2d.append((x, y, point3d_id))
                        
                        image_points[image_id] = points_2d
            i += 1
        
        print(f"✅ Found {len(images)} images")
        total_2d_points = sum(len(pts) for pts in image_points.values())
        print(f"✅ Found {total_2d_points} 2D point observations")
        
    except Exception as e:
        issues.append(f"❌ Error reading images.txt: {e}")
    
    # Validate points3D.txt
    print("\n🎯 Validating points3D.txt...")
    try:
        points3d = {}
        with open(points3d_file, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) < 8:
                    issues.append(f"❌ Point3D line {line_num}: Insufficient data")
                    continue
                
                point3d_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                error = float(parts[7])
                
                # Check for finite coordinates
                if not all(np.isfinite([x, y, z, error])):
                    issues.append(f"❌ Point3D {point3d_id}: Non-finite values")
                
                # Check color values
                if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                    issues.append(f"❌ Point3D {point3d_id}: Invalid color values ({r}, {g}, {b})")
                
                # Parse track data
                track_data = parts[8:]
                if len(track_data) % 2 != 0:
                    issues.append(f"❌ Point3D {point3d_id}: Track data not even length: {len(track_data)}")
                else:
                    track_length = len(track_data) // 2
                    if track_length < 2:
                        issues.append(f"❌ Point3D {point3d_id}: Track too short ({track_length} observations)")
                    
                    for i in range(track_length):
                        image_id = int(track_data[i*2])
                        point2d_idx = int(track_data[i*2 + 1])
                        
                        # Check if image exists
                        if image_id not in images:
                            issues.append(f"❌ Point3D {point3d_id}: References non-existent image {image_id}")
                        
                        # Check if 2D point index is valid
                        if image_id in image_points:
                            if point2d_idx >= len(image_points[image_id]):
                                issues.append(f"❌ Point3D {point3d_id}: Invalid 2D point index {point2d_idx} for image {image_id}")
                
                points3d[point3d_id] = (x, y, z, r, g, b, error, track_data)
        
        print(f"✅ Found {len(points3d)} 3D points")
        
        # Check track consistency
        print("\n🔗 Checking track consistency...")
        for point3d_id, (x, y, z, r, g, b, error, track_data) in points3d.items():
            track_length = len(track_data) // 2
            for i in range(track_length):
                image_id = int(track_data[i*2])
                point2d_idx = int(track_data[i*2 + 1])
                
                if image_id in image_points and point2d_idx < len(image_points[image_id]):
                    _, _, referenced_point3d_id = image_points[image_id][point2d_idx]
                    if referenced_point3d_id != point3d_id:
                        issues.append(f"❌ Track inconsistency: Point3D {point3d_id} references image {image_id} point {point2d_idx}, but that 2D point references Point3D {referenced_point3d_id}")
        
    except Exception as e:
        issues.append(f"❌ Error reading points3D.txt: {e}")
    
    # Summary
    print(f"\n📊 Validation Summary:")
    if not issues:
        print("🎉 All checks passed! The reconstruction should be valid for COLMAP.")
        
        # Print statistics
        avg_track_length = sum(len(track_data) // 2 for _, _, _, _, _, _, _, track_data in points3d.values()) / len(points3d) if points3d else 0
        print(f"📈 Statistics:")
        print(f"   - Cameras: {len(cameras)}")
        print(f"   - Images: {len(images)}")
        print(f"   - 3D Points: {len(points3d)}")
        print(f"   - Average track length: {avg_track_length:.1f}")
        
    else:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"   {issue}")
    
    return issues

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_colmap.py <sparse_dir>")
        sys.exit(1)
    
    sparse_dir = sys.argv[1]
    issues = validate_colmap_reconstruction(sparse_dir)
    
    sys.exit(0 if not issues else 1) 