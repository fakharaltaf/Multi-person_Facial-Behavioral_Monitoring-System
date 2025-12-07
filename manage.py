"""
Project setup and management CLI
Convenience script for common operations
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display result"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def setup_environment(args):
    """Set up Python environment"""
    print("Setting up environment...")
    
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print(f"\n✗ Failed: {desc}")
            return False
    
    print("\n✓ Environment setup complete!")
    return True


def verify_setup(args):
    """Verify system setup"""
    return run_command("python src/utils/verify_setup.py", "Verifying setup")


def download_models(args):
    """Download models"""
    if args.model:
        cmd = f"python src/utils/download_models.py --model {args.model}"
    elif args.all:
        cmd = "python src/utils/download_models.py --all"
    else:
        cmd = "python src/utils/download_models.py --list"
    
    return run_command(cmd, "Model downloader")


def run_tests(args):
    """Run tests"""
    if args.stage:
        test_file = f"tests/test_stage{args.stage}*.py"
        cmd = f"python -m pytest {test_file} -v"
    elif args.all:
        cmd = "python -m pytest tests/ -v"
    else:
        cmd = "python -m pytest tests/ -v --maxfail=1"
    
    return run_command(cmd, "Running tests")


def clean_outputs(args):
    """Clean output directories"""
    import shutil
    
    print("\nCleaning output directories...")
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "outputs"
    
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
                print(f"  Deleted: {item.name}")
            elif item.is_dir() and not item.name.startswith('.'):
                shutil.rmtree(item)
                print(f"  Deleted directory: {item.name}")
    
    print("✓ Cleanup complete!")
    return True


def show_status(args):
    """Show project status"""
    import yaml
    
    project_root = Path(__file__).parent.parent
    
    print("\n" + "="*60)
    print("PROJECT STATUS")
    print("="*60)
    
    # Check environment
    print("\n📦 Environment:")
    result = subprocess.run(
        ["pip", "list", "--format=columns"],
        capture_output=True,
        text=True
    )
    key_packages = ["numpy", "opencv-python", "onnxruntime", "onnx"]
    for line in result.stdout.split('\n'):
        for pkg in key_packages:
            if pkg in line.lower():
                print(f"  {line}")
    
    # Check models
    print("\n🤖 Models:")
    models_dir = project_root / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.onnx"))
        if model_files:
            for model in model_files:
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"  ✓ {model.name} ({size_mb:.1f} MB)")
        else:
            print("  ✗ No models downloaded")
    
    # Check configuration
    print("\n⚙️  Configuration:")
    config_file = project_root / "config" / "system_config.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  Device: {config['system']['device']}")
        print(f"  Max people: {config['system']['max_people']}")
        print(f"  Target FPS: {config['system']['target_fps']}")
        print(f"  Detection model: {config['detection']['model_type']}")
        print(f"  Recognition model: {config['recognition']['model_type']}")
        print(f"  Tracker: {config['tracking']['tracker_type']}")
    
    # Show roadmap progress
    print("\n📋 Development Progress:")
    stages = [
        ("Stage 0", "Foundation", "✅ Complete"),
        ("Stage 1", "Face Detection", "🔜 Next"),
        ("Stage 2", "Face Recognition", "⏳ Pending"),
        ("Stage 3", "Multi-Object Tracking", "⏳ Pending"),
        ("Stage 4", "Behavioral Analysis", "⏳ Pending"),
        ("Stage 5", "Integration", "⏳ Pending"),
    ]
    for stage, name, status in stages:
        print(f"  {stage}: {name:25s} {status}")
    
    print("\n" + "="*60)
    print("\nFor detailed roadmap: see DEVELOPMENT_ROADMAP.md")
    print("For quick start: see QUICKSTART.md")
    print("\n" + "="*60)
    
    return True


def main():
    """Main CLI"""
    parser = argparse.ArgumentParser(
        description="Multi-Person Tracking System - Project Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup
  python manage.py setup              # Install dependencies
  python manage.py verify             # Verify installation
  
  # Models
  python manage.py models --all       # Download all models
  python manage.py models --list      # List available models
  python manage.py models --model detection/retinaface_resnet50
  
  # Testing
  python manage.py test               # Run all tests
  python manage.py test --stage 1     # Run stage 1 tests
  
  # Utilities
  python manage.py status             # Show project status
  python manage.py clean              # Clean outputs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup
    parser_setup = subparsers.add_parser('setup', help='Set up environment')
    parser_setup.set_defaults(func=setup_environment)
    
    # Verify
    parser_verify = subparsers.add_parser('verify', help='Verify setup')
    parser_verify.set_defaults(func=verify_setup)
    
    # Models
    parser_models = subparsers.add_parser('models', help='Manage models')
    parser_models.add_argument('--all', action='store_true', help='Download all models')
    parser_models.add_argument('--list', action='store_true', help='List available models')
    parser_models.add_argument('--model', type=str, help='Download specific model')
    parser_models.set_defaults(func=download_models)
    
    # Test
    parser_test = subparsers.add_parser('test', help='Run tests')
    parser_test.add_argument('--stage', type=int, help='Run specific stage tests')
    parser_test.add_argument('--all', action='store_true', help='Run all tests')
    parser_test.set_defaults(func=run_tests)
    
    # Clean
    parser_clean = subparsers.add_parser('clean', help='Clean output directories')
    parser_clean.set_defaults(func=clean_outputs)
    
    # Status
    parser_status = subparsers.add_parser('status', help='Show project status')
    parser_status.set_defaults(func=show_status)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Run command
    try:
        success = args.func(args)
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
