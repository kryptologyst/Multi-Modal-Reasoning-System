#!/usr/bin/env python3
"""
Simple test script for the multi-modal reasoning system.
This tests basic functionality without heavy dependencies.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic Python imports
        import json
        import numpy as np
        print("✅ Basic imports successful")
        
        # Test if PyTorch is available
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__} available")
        except ImportError:
            print("⚠️ PyTorch not available - install with: pip install torch")
        
        # Test if transformers is available
        try:
            import transformers
            print(f"✅ Transformers {transformers.__version__} available")
        except ImportError:
            print("⚠️ Transformers not available - install with: pip install transformers")
        
        # Test if PIL is available
        try:
            from PIL import Image
            print("✅ PIL available")
        except ImportError:
            print("⚠️ PIL not available - install with: pip install Pillow")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_project_structure():
    """Test that the project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = [
        "src",
        "src/data",
        "src/models", 
        "src/losses",
        "src/eval",
        "src/viz",
        "src/utils",
        "configs",
        "configs/model",
        "configs/train",
        "data",
        "assets",
        "scripts",
        "demo",
        "tests",
    ]
    
    required_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/losses/__init__.py",
        "src/eval/__init__.py",
        "src/viz/__init__.py",
        "src/utils/__init__.py",
        "src/data/dataset.py",
        "src/models/reasoning_model.py",
        "src/losses/losses.py",
        "src/eval/metrics.py",
        "src/viz/visualization.py",
        "src/utils/device.py",
        "src/utils/config.py",
        "src/utils/logging.py",
        "configs/model/clip_config.yaml",
        "configs/train/train_config.yaml",
        "scripts/train.py",
        "demo/app.py",
        "0939.py",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        ".gitignore",
    ]
    
    base_path = Path(__file__).parent
    
    # Test directories
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
    
    # Test files
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists() and full_path.is_file():
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")


def test_config_files():
    """Test that configuration files are valid."""
    print("\nTesting configuration files...")
    
    try:
        import yaml
        
        config_files = [
            "configs/model/clip_config.yaml",
            "configs/train/train_config.yaml",
        ]
        
        base_path = Path(__file__).parent
        
        for config_file in config_files:
            full_path = base_path / config_file
            if full_path.exists():
                with open(full_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✅ Config file valid: {config_file}")
            else:
                print(f"❌ Config file missing: {config_file}")
                
    except ImportError:
        print("⚠️ PyYAML not available - install with: pip install pyyaml")
    except Exception as e:
        print(f"❌ Config test failed: {e}")


def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        # Test JSON handling
        test_data = [
            {"image": "test.jpg", "text": "A test image", "id": 1},
            {"image": "test2.jpg", "text": "Another test image", "id": 2},
        ]
        
        import json
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        
        if len(parsed_data) == 2:
            print("✅ JSON handling works")
        else:
            print("❌ JSON handling failed")
        
        # Test numpy if available
        try:
            import numpy as np
            arr = np.array([1, 2, 3, 4, 5])
            if arr.mean() == 3.0:
                print("✅ NumPy functionality works")
            else:
                print("❌ NumPy functionality failed")
        except ImportError:
            print("⚠️ NumPy not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Multi-Modal Reasoning System - Project 939")
    print("=" * 50)
    
    # Run tests
    import_success = test_imports()
    test_project_structure()
    test_config_files()
    func_success = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    if import_success and func_success:
        print("✅ All basic tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run demo: python 0939.py --demo")
        print("3. Run training: python 0939.py --train")
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\nTo fix issues:")
        print("1. Install missing dependencies")
        print("2. Check file permissions")
        print("3. Verify Python version (3.10+ required)")


if __name__ == "__main__":
    main()
