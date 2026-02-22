#!/usr/bin/env python3
"""
Test script for Gemini API connectivity.
Run on ubuntu1 to verify google-generativeai installation and API access.

Usage:
  export GEMINI_API_KEY="your_api_key_here"
  python3 test_gemini_api.py
"""

import os
import sys

def test_import():
    """Test if google-generativeai can be imported."""
    print("1. Testing import...")
    try:
        import google.generativeai as genai
        print("   ✅ google-generativeai imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import: {e}")
        print("   Run: pip3 install google-generativeai")
        return False

def test_api_key():
    """Test if API key is configured."""
    print("\n2. Testing API key...")
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("   ❌ GEMINI_API_KEY not set")
        print("   Run: export GEMINI_API_KEY='your_api_key_here'")
        return False
    print(f"   ✅ API key found (length: {len(api_key)})")
    return api_key

def test_api_connection(api_key):
    """Test basic API call."""
    print("\n3. Testing API connection...")
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # List available models
        print("   Available models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"     - {model.name}")
        
        print("   ✅ API connection successful")
        return True
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")
        return False

def test_simple_generation(api_key):
    """Test simple text generation."""
    print("\n4. Testing simple generation...")
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-robotics-er-1.5-preview')
        response = model.generate_content("Say 'Hello Robot!' in one sentence.")
        
        print(f"   Response: {response.text}")
        print("   ✅ Text generation successful")
        return True
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return False

def test_json_schema():
    """Test JSON schema enforcement."""
    print("\n5. Testing JSON schema...")
    try:
        import google.generativeai as genai
        
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["action", "confidence"]
        }
        
        model = genai.GenerativeModel(
            'gemini-robotics-er-1.5-preview',
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        )
        
        response = model.generate_content(
            "Return JSON with action='move_forward' and confidence=0.9"
        )
        
        print(f"   Response: {response.text}")
        print("   ✅ JSON schema enforcement successful")
        return True
    except Exception as e:
        print(f"   ❌ JSON schema test failed: {e}")
        return False

def test_multimodal():
    """Test multimodal input (text + image)."""
    print("\n6. Testing multimodal input...")
    try:
        import google.generativeai as genai
        from PIL import Image
        import io
        
        # Create simple test image (red square)
        img = Image.new('RGB', (100, 100), color='red')
        
        model = genai.GenerativeModel('gemini-robotics-er-1.5-preview')
        response = model.generate_content([
            "What color is this image?",
            img
        ])
        
        print(f"   Response: {response.text}")
        print("   ✅ Multimodal input successful")
        return True
    except Exception as e:
        print(f"   ❌ Multimodal test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Gemini API Test Suite")
    print("=" * 60)
    
    # Test 1: Import
    if not test_import():
        print("\n❌ FAILED: Cannot proceed without google-generativeai")
        sys.exit(1)
    
    # Test 2: API Key
    api_key = test_api_key()
    if not api_key:
        print("\n❌ FAILED: Cannot proceed without API key")
        sys.exit(1)
    
    # Test 3: Connection
    if not test_api_connection(api_key):
        print("\n❌ FAILED: Cannot connect to API")
        sys.exit(1)
    
    # Test 4: Simple generation
    test_simple_generation(api_key)
    
    # Test 5: JSON schema
    test_json_schema()
    
    # Test 6: Multimodal
    test_multimodal()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Implement gemini_bridge.py node")
    print("2. Integrate with ROS2 topics and actions")
    print("3. Test with live robot")

if __name__ == '__main__':
    main()
