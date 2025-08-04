#!/usr/bin/env python3
"""
Test script to verify dependencies are working correctly in the Docker environment.

This script imports key dependencies and performs basic operations to ensure
the environment is set up correctly.

Usage:
    python tools/test_dependencies.py
"""

import sys
import os

print("Testing dependencies...")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Test numpy
try:
    import numpy as np
    print(f"✅ NumPy imported successfully - version: {np.__version__}")
    
    # Basic numpy operations
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4, 6, 8, 10])
    
    print(f"   Array a: {a}")
    print(f"   Array b: {b}")
    print(f"   Sum: {a + b}")
    print(f"   Dot product: {np.dot(a, b)}")
    print(f"   Mean of a: {np.mean(a)}")
    
    # Test matrix operations
    matrix = np.random.rand(3, 3)
    print(f"   Random 3x3 matrix shape: {matrix.shape}")
    print(f"   Matrix determinant: {np.linalg.det(matrix):.4f}")
    print()
    
except ImportError as e:
    print(f"❌ Failed to import NumPy: {e}")
    sys.exit(1)

# Test torch
try:
    import torch
    print(f"✅ PyTorch imported successfully - version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.current_device()}")
        print(f"   GPU name: {torch.cuda.get_device_name()}")
    
    # Basic torch operations
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    print(f"   Tensor x: {x}")
    print(f"   Tensor y: {y}")
    print(f"   x + y: {x + y}")
    print(f"   x * y: {x * y}")
    
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        result_gpu = x_gpu + y_gpu
        print(f"   GPU computation result: {result_gpu.cpu()}")
    print()
    
except ImportError as e:
    print(f"❌ Failed to import PyTorch: {e}")

# Test other key dependencies
dependencies = [
    ('PIL', 'Pillow'),
    ('cv2', 'OpenCV'),
    ('matplotlib', 'Matplotlib'),
    ('scipy', 'SciPy'),
    ('trimesh', 'Trimesh'),
    ('gradio', 'Gradio'),
]

for module_name, package_name in dependencies:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name} imported successfully - version: {version}")
    except ImportError as e:
        print(f"❌ Failed to import {package_name}: {e}")

print()

# Test project-specific imports
print("Testing project-specific imports...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from streamvggt.models.streamvggt import StreamVGGT
    print("✅ StreamVGGT model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import StreamVGGT: {e}")

try:
    from streamvggt.utils.load_fn import load_and_preprocess_images
    print("✅ load_and_preprocess_images imported successfully")
except ImportError as e:
    print(f"❌ Failed to import load_and_preprocess_images: {e}")

try:
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    print("✅ pose_encoding_to_extri_intri imported successfully")
except ImportError as e:
    print(f"❌ Failed to import pose_encoding_to_extri_intri: {e}")

try:
    from visual_util import predictions_to_glb
    print("✅ predictions_to_glb imported successfully")
except ImportError as e:
    print(f"❌ Failed to import predictions_to_glb: {e}")

print()
print("Dependency test completed!")

# Basic computation test
print("\nPerforming basic computation test...")
try:
    # Create a simple computation using multiple libraries
    import numpy as np
    import torch
    
    # NumPy computation
    np_data = np.random.randn(1000, 100)
    np_mean = np.mean(np_data)
    np_std = np.std(np_data)
    
    # PyTorch computation
    torch_data = torch.from_numpy(np_data).float()
    if torch.cuda.is_available():
        torch_data = torch_data.cuda()
    
    torch_mean = torch.mean(torch_data)
    torch_std = torch.std(torch_data)
    
    print(f"NumPy - Mean: {np_mean:.4f}, Std: {np_std:.4f}")
    print(f"PyTorch - Mean: {torch_mean.cpu().item():.4f}, Std: {torch_std.cpu().item():.4f}")
    print("✅ Basic computation test passed!")
    
except Exception as e:
    print(f"❌ Basic computation test failed: {e}")

print("\n🎉 All tests completed!") 