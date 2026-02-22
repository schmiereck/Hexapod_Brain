#!/usr/bin/env python3
"""
Test script for Gemini API connectivity (google.genai).
Run on ubuntu1 to verify google-genai installation and API access.

Usage:
  export GEMINI_API_KEY="your_api_key_here"
  python3 test_gemini_api_new.py
"""

import os
import sys
import json

def test_import():
    """Test if google.genai can be imported."""
    print("1. Testing import...")
    try:
        from google import genai
        print("   ✅ google.genai imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import: {e}")
        print("   Run: pip3 install google-genai")
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
        from google import genai
        client = genai.Client(api_key=api_key)
        
        # List available models
        print("   Available models:")
        models = client.models.list()
        for model in models:
            print(f"     - {model.name}")
        
        print("   ✅ API connection successful")
        return client
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")
        return None

def test_simple_generation(client):
    """Test simple text generation."""
    print("\n4. Testing simple generation...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Hello Robot!'
        )
        print(f"   Response: {response.text}")
        print("   ✅ Text generation successful")
        return True
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return False

def test_json_schema(client):
    """Test JSON schema enforcement."""
    print("\n5. Testing JSON schema...")
    
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["action", "confidence"]
    }
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Return an action to move forward with high confidence',
            config={
                "response_mime_type": "application/json",
                "response_schema": schema
            }
        )
        result = json.loads(response.text)
        print(f"   Response: {json.dumps(result, indent=2)}")
        print("   ✅ JSON schema enforcement successful")
        return True
    except Exception as e:
        print(f"   ❌ JSON schema test failed: {e}")
        return False

def test_multimodal_input(client):
    """Test multimodal input (image + text)."""
    print("\n6. Testing multimodal input...")
    try:
        from PIL import Image
        import io
        
        # Create test image (red square)
        img = Image.new('RGB', (100, 100), color='red')
        print("   ✅ Test image created")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "What color is this image? Answer in one word.",
                img
            ]
        )
        print(f"   Response: {response.text}")
        print("   ✅ Multimodal input successful")
        return True
    except Exception as e:
        print(f"   ❌ Multimodal test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Gemini API Test Suite (google.genai)")
    print("=" * 60)
    
    # Test 1: Import
    if not test_import():
        sys.exit(1)
    
    # Test 2: API Key
    api_key = test_api_key()
    if not api_key:
        sys.exit(1)
    
    # Test 3: API Connection
    client = test_api_connection(api_key)
    if not client:
        sys.exit(1)
    
    # Test 4: Simple Generation
    if not test_simple_generation(client):
        sys.exit(1)
    
    # Test 5: JSON Schema
    if not test_json_schema(client):
        sys.exit(1)
    
    # Test 6: Multimodal
    if not test_multimodal_input(client):
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test with gemini_bridge.py node")
    print("2. Integrate with ROS2 topics and actions")
    print("3. Test with live robot")

if __name__ == '__main__':
    main()
