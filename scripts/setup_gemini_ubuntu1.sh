#!/bin/bash
# Setup script for Gemini API on ubuntu1
# Run this script on ubuntu1 after SSHing in

set -e  # Exit on error

echo "=========================================="
echo "Gemini API Setup for ubuntu1"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
PYTHON_VERSION=$(python3 --version)
echo "   $PYTHON_VERSION"

# Install google-generativeai
echo ""
echo "2. Installing google-generativeai..."
pip3 install google-generativeai pillow

# Verify installation
echo ""
echo "3. Verifying installation..."
python3 -c "import google.generativeai as genai; print('   ✅ google-generativeai imported successfully')"
python3 -c "import PIL; print('   ✅ Pillow imported successfully')"

# Check for API key
echo ""
echo "4. Checking for API key..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "   ❌ GEMINI_API_KEY not set"
    echo ""
    echo "   To set API key, add to ~/.bashrc:"
    echo "   export GEMINI_API_KEY='your_api_key_here'"
    echo ""
    echo "   Then run: source ~/.bashrc"
else
    echo "   ✅ GEMINI_API_KEY is set"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set GEMINI_API_KEY in ~/.bashrc (if not done)"
echo "2. Run test: python3 ~/Hexapod_Brain/scripts/test_gemini_api.py"
echo "3. If tests pass, proceed with gemini_bridge.py implementation"
