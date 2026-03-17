#!/bin/bash
# Install ROCm system-wide on Ubuntu 25.10 for AMD Strix Halo (gfx1151)
# Run with sudo or as root. Reboot required after.
# Rockwood Lab LLC — https://rockwoodlab.com

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (sudo bash install_rocm.sh)"
    exit 1
fi

ROCM_VERSION="6.4"  # ROCm 6.4 = runtime version 7.10.x
UBUNTU_CODENAME="oracular"  # Ubuntu 25.10 — update if needed

echo "=== Installing ROCm $ROCM_VERSION for Ubuntu 25.10 ==="
echo ""

# ── AMD GPU repo ─────────────────────────────────────────────────────────────

echo "--- Adding AMD GPU repository ---"
apt-get update -qq
apt-get install -y wget gnupg curl

# AMD ROCm repo key
wget -qO /etc/apt/keyrings/rocm.gpg \
    https://repo.radeon.com/rocm/rocm.gpg.key

# Add repo — Ubuntu 25.10 (oracular) may need mantic/noble fallback
# Try oracular first, fall back to noble if packages not found
cat > /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
    https://repo.radeon.com/rocm/apt/$ROCM_VERSION $UBUNTU_CODENAME main
EOF

apt-get update -qq || {
    echo "oracular repo failed, trying noble..."
    cat > /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
    https://repo.radeon.com/rocm/apt/$ROCM_VERSION noble main
EOF
    apt-get update -qq
}

# ── Install ROCm packages ─────────────────────────────────────────────────────

echo "--- Installing ROCm core packages ---"
apt-get install -y \
    rocm-dev \
    rocm-libs \
    rocminfo \
    rocm-smi-lib \
    hipcc

# ── PATH and LD_LIBRARY_PATH ──────────────────────────────────────────────────

echo "--- Setting up ROCm environment ---"

cat > /etc/profile.d/rocm.sh << 'EOF'
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=11.5.1
EOF

echo "ROCm environment written to /etc/profile.d/rocm.sh"

# ── User groups ───────────────────────────────────────────────────────────────

echo "--- Adding current user to render and video groups ---"
REAL_USER="${SUDO_USER:-$USER}"
usermod -aG render,video "$REAL_USER"
echo "Added $REAL_USER to render and video groups"

# ── Verify ────────────────────────────────────────────────────────────────────

echo ""
echo "=== Installation complete ==="
echo ""
echo "IMPORTANT: You must reboot before ROCm will be fully available."
echo ""
echo "After reboot, verify with:"
echo "  rocm-smi        — should show your GPU"
echo "  rocminfo        — should list gfx1151"
echo ""
echo "Then run setup.sh from the autoresearch repo root."
