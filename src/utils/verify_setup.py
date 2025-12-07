"""
Verify system setup and dependencies
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_package(package_name, import_name=None):
    """Check if package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name} ({version})")
        return True
    except ImportError:
        print(f"  ✗ {package_name} (not installed)")
        return False


def check_onnxruntime():
    """Check ONNXRuntime and DirectML support"""
    print("\nChecking ONNXRuntime...")
    
    try:
        import onnxruntime as ort
        print(f"  ✓ ONNXRuntime ({ort.__version__})")
        
        providers = ort.get_available_providers()
        print(f"\n  Available providers:")
        for provider in providers:
            print(f"    - {provider}")
        
        if 'DmlExecutionProvider' in providers:
            print(f"  ✓ DirectML support available (AMD GPU acceleration)")
            return True
        else:
            print(f"  ⚠ DirectML not available (will use CPU)")
            print(f"    Install: pip install onnxruntime-directml")
            return False
            
    except ImportError:
        print(f"  ✗ ONNXRuntime not installed")
        return False


def check_opencv():
    """Check OpenCV installation"""
    print("\nChecking OpenCV...")
    
    try:
        import cv2
        print(f"  ✓ OpenCV ({cv2.__version__})")
        
        # Check build info for optimization flags
        build_info = cv2.getBuildInformation()
        if 'AVX' in build_info or 'AVX2' in build_info:
            print(f"  ✓ Optimized build detected")
        
        return True
        
    except ImportError:
        print(f"  ✗ OpenCV not installed")
        return False


def check_project_structure():
    """Check project directory structure"""
    print("\nChecking project structure...")
    
    project_root = Path(__file__).parent.parent.parent
    
    required_dirs = [
        "config",
        "models",
        "src/detection",
        "src/recognition",
        "src/tracking",
        "src/behavior",
        "src/utils",
        "src/visualization",
        "data/outputs",
        "tests"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def check_models():
    """Check if models are downloaded"""
    print("\nChecking models...")
    
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    
    required_models = [
        "retinaface_resnet50.onnx",
        "arcface_resnet100.onnx"
    ]
    
    any_model = False
    for model_name in required_models:
        model_path = models_dir / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {model_name} ({size_mb:.1f} MB)")
            any_model = True
        else:
            print(f"  ✗ {model_name} (not downloaded)")
    
    if not any_model:
        print(f"\n  To download models, run:")
        print(f"    python src/utils/download_models.py --all")
    
    return any_model


def main():
    """Run all checks"""
    print("=" * 60)
    print("SYSTEM SETUP VERIFICATION")
    print("Multi-Person Tracking System")
    print("=" * 60)
    
    checks = []
    
    # Python version
    checks.append(("Python Version", check_python_version()))
    
    # Core packages
    print("\nChecking core packages...")
    checks.append(("NumPy", check_package("numpy")))
    checks.append(("OpenCV", check_opencv()))
    checks.append(("ONNX", check_package("onnx")))
    checks.append(("ONNXRuntime", check_onnxruntime()))
    
    # Additional packages
    print("\nChecking additional packages...")
    checks.append(("SciPy", check_package("scipy")))
    checks.append(("scikit-learn", check_package("sklearn", "sklearn")))
    checks.append(("Pandas", check_package("pandas")))
    checks.append(("Matplotlib", check_package("matplotlib")))
    checks.append(("PIL", check_package("Pillow", "PIL")))
    checks.append(("YAML", check_package("PyYAML", "yaml")))
    checks.append(("FilterPy", check_package("filterpy")))
    
    # Project structure
    checks.append(("Project Structure", check_project_structure()))
    
    # Models
    checks.append(("Models", check_models()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All checks passed! System is ready.")
    else:
        print("\n⚠ Some checks failed. Please install missing dependencies.")
        print("\nTo install all requirements:")
        print("  pip install -r requirements.txt")
        print("\nTo download models:")
        print("  python src/utils/download_models.py --all")
    
    print("\n" + "=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
