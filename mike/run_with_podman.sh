#!/bin/bash
# Helper script to run SageMaker local mode with Podman

echo "=========================================="
echo "SageMaker Local Mode with Podman"
echo "=========================================="
echo ""

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    echo "ERROR: Podman is not installed!"
    echo "Please install it first:"
    echo "  Fedora/RHEL: sudo dnf install podman"
    echo "  Debian/Ubuntu: sudo apt-get install podman"
    exit 1
fi

echo "✓ Podman is installed"

# Check if Podman socket is running
if systemctl --user is-active --quiet podman.socket; then
    echo "✓ Podman socket is running"
else
    echo "⚠ Podman socket is not running. Starting it..."
    systemctl --user enable --now podman.socket
    sleep 2
    if systemctl --user is-active --quiet podman.socket; then
        echo "✓ Podman socket started successfully"
    else
        echo "ERROR: Could not start Podman socket"
        exit 1
    fi
fi

# Set DOCKER_HOST environment variable
export DOCKER_HOST=unix:///run/user/$UID/podman/podman.sock
echo "✓ Set DOCKER_HOST=$DOCKER_HOST"
echo ""

# Test Podman connectivity
echo "Testing Podman connectivity..."
if podman ps &> /dev/null; then
    echo "✓ Podman is accessible"
else
    echo "ERROR: Cannot connect to Podman"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running invoke_local.py with Podman"
echo "=========================================="
echo ""

# Run the script
python invoke_local.py

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
