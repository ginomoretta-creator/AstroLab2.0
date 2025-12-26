#!/bin/bash
# Build script for bundling Python backend with PyInstaller
# Run this before building the Electron app for distribution

echo "=== ASL-Sandbox Python Backend Bundler ==="
echo ""

# Check if PyInstaller is installed
if ! pip show pyinstaller > /dev/null 2>&1; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Navigate to the desktop-app/python directory
cd "$(dirname "$0")"

# Run PyInstaller
echo "Building Python backend..."
pyinstaller backend.spec --noconfirm --clean

# Copy the output to the Electron app resources
echo "Copying to Electron resources..."
mkdir -p ../resources/backend
cp -r dist/asl-sandbox-backend* ../resources/backend/

echo ""
echo "=== Build Complete ==="
echo "Backend executable is in: resources/backend/"
