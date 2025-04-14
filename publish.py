#!/usr/bin/env python
"""
Script to publish Neural-Scope to PyPI.
"""

import os
import sys
import shutil
import subprocess
# from pathlib import Path  # Not used

def clean_build_dirs():
    """Clean build directories."""
    print("Cleaning build directories...")

    # Remove build directories
    for dir_name in ["build", "dist", "neural_scope.egg-info"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    print("Build directories cleaned.")

def build_package():
    """Build the package."""
    print("Building package...")

    # Use PyPI-specific README
    if os.path.exists("README_PYPI.md") and os.path.exists("README.md"):
        print("Using PyPI-specific README...")
        shutil.copy("README_PYPI.md", "README.md.bak")
        shutil.copy("README_PYPI.md", "README.md")

    try:
        # Build the package
        subprocess.run([sys.executable, "setup.py", "sdist", "bdist_wheel"], check=True)
        print("Package built successfully.")
    finally:
        # Restore original README
        if os.path.exists("README.md.bak"):
            print("Restoring original README...")
            shutil.move("README.md.bak", "README.md")

def check_package():
    """Check the package with twine."""
    print("Checking package...")

    # Check the package
    subprocess.run(["twine", "check", "dist/*"], check=True)

    print("Package check passed.")

def publish_to_pypi():
    """Publish the package to PyPI."""
    print("Publishing to PyPI...")

    # Publish to PyPI
    subprocess.run(["twine", "upload", "dist/*"], check=True)

    print("Package published to PyPI successfully.")

def publish_to_test_pypi():
    """Publish the package to Test PyPI."""
    print("Publishing to Test PyPI...")

    # Publish to Test PyPI
    subprocess.run(["twine", "upload", "--repository-url", "https://test.pypi.org/legacy/", "dist/*"], check=True)

    print("Package published to Test PyPI successfully.")

def main():
    """Run the publish script."""
    # Parse arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "test"

    # Clean build directories
    clean_build_dirs()

    # Build package
    build_package()

    # Check package
    check_package()

    # Publish package
    if command == "test":
        publish_to_test_pypi()
    elif command == "prod":
        publish_to_pypi()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python publish.py [test|prod]")
        sys.exit(1)

    print("Done!")

if __name__ == "__main__":
    main()
