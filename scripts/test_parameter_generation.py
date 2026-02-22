#!/usr/bin/env python3
"""
Test script to debug Gemini parameter generation issue.
Tests different prompts and schemas to find why parameters={} is always empty.
"""

import os
import json
import google.generativeai as genai
from PIL import Image
import io

# Configure API
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("ERROR: GEMINI_API_KEY not set")
    exit(1)

genai.configure(api_key=api_key)

# Create a simple test image (red square with white circle - like a bottle)
def create_test_image():
    img = Image.new('RGB', (640, 480), color='gray')
    # Draw a simple "bottle" shape (white rectangle)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Bottle on the left side (x=120)
    draw.rectangle([100, 200, 140, 350], fill='white', outline='black', width=2)
    return img

test_image = create_test_image()
print("✅ Test image created (bottle on left at x=120)")

# Test detection data
test_detections = [
    {
        "class_id": "bottle",
        "score": 0.89,
        "bbox": {
            "center": {"x": 120, "y": 275},
            "size_x": 40,
            "size_y": 150
        }
    }
]

print("\n" + "="*60)
print("TEST 1: Original Schema (with description)")
print("="*60)

schema_v1 = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "object",
            "properties": {
                "observation": {"type": "string"},
                "goal_status": {"type": "string"},
                "affordance_check": {"type": "string"},
                "safety_rating": {"type": "string", "enum": ["high", "medium", "low"]}
            },
            "required": ["observation", "goal_status", "affordance_check", "safety_rating"]
        },
        "action": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["linear_move", "rotate", "head_position", "wait"]},
                "parameters": {
                    "type": "object",
                    "description": "MUST include all required fields. For rotate: angle_degrees (number), speed (number)."
                }
            },
            "required": ["type", "parameters"]
        },
        "explanation": {"type": "string"}
    },
    "required": ["reasoning", "action", "explanation"]
}

prompt_v1 = f"""
**Goal**: Center the bottle in view

**Image**: 640x480 pixels, image center is x=320

**Detected Objects**: 
{json.dumps(test_detections, indent=2)}

The bottle is at x=120 (LEFT of center). You MUST rotate RIGHT (positive angle) to center it.

**CRITICAL**: Your response MUST include the "parameters" field with ALL required values!

For rotate action, you MUST include:
- angle_degrees: float (POSITIVE for right turn, NEGATIVE for left turn)
- speed: float (0-100)

Example for this situation:
```json
{{
  "action": {{
    "type": "rotate",
    "parameters": {{
      "angle_degrees": 25.0,
      "speed": 40.0
    }}
  }}
}}
```

What action should I take?
"""

try:
    model_v1 = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema_v1
        }
    )
    
    response = model_v1.generate_content([prompt_v1, test_image])
    result = json.loads(response.text)
    
    print("\nResponse:")
    print(json.dumps(result, indent=2))
    
    params = result.get('action', {}).get('parameters', {})
    if params:
        print(f"\n✅ SUCCESS! Parameters: {params}")
    else:
        print(f"\n❌ FAILED! Parameters empty: {params}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("TEST 2: Explicit Parameter Schema (properties defined)")
print("="*60)

schema_v2 = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "object",
            "properties": {
                "observation": {"type": "string"},
                "safety_rating": {"type": "string", "enum": ["high", "medium", "low"]}
            },
            "required": ["observation", "safety_rating"]
        },
        "action": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["rotate"]},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "angle_degrees": {"type": "number"},
                        "speed": {"type": "number"}
                    },
                    "required": ["angle_degrees", "speed"]
                }
            },
            "required": ["type", "parameters"]
        }
    },
    "required": ["reasoning", "action"]
}

prompt_v2 = f"""
Bottle detected at x=120 (left of center x=320).

Rotate right to center it. Return JSON with:
- action.type: "rotate"
- action.parameters.angle_degrees: positive float (20-40)
- action.parameters.speed: float (30-50)

Detected objects: {json.dumps(test_detections)}
"""

try:
    model_v2 = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema_v2
        }
    )
    
    response = model_v2.generate_content([prompt_v2, test_image])
    result = json.loads(response.text)
    
    print("\nResponse:")
    print(json.dumps(result, indent=2))
    
    params = result.get('action', {}).get('parameters', {})
    if params and 'angle_degrees' in params:
        print(f"\n✅ SUCCESS! Parameters: {params}")
    else:
        print(f"\n❌ FAILED! Parameters: {params}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("TEST 3: Minimal Schema (see if required works)")
print("="*60)

schema_v3 = {
    "type": "object",
    "properties": {
        "angle_degrees": {"type": "number"},
        "speed": {"type": "number"}
    },
    "required": ["angle_degrees", "speed"]
}

prompt_v3 = "Bottle at x=120, center is x=320. Return angle_degrees (positive float) and speed (float) to rotate right."

try:
    model_v3 = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema_v3
        }
    )
    
    response = model_v3.generate_content([prompt_v3, test_image])
    result = json.loads(response.text)
    
    print("\nResponse:")
    print(json.dumps(result, indent=2))
    
    if 'angle_degrees' in result and 'speed' in result:
        print(f"\n✅ SUCCESS! Got values: angle={result['angle_degrees']}, speed={result['speed']}")
    else:
        print(f"\n❌ FAILED! Missing fields")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("TEST 4: Without JSON Schema (plain text, then parse)")
print("="*60)

try:
    model_v4 = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt_v4 = """
Bottle detected at x=120 (left of center x=320 in 640px wide image).

Return ONLY a JSON object with this EXACT structure:
{
  "action_type": "rotate",
  "angle_degrees": <positive number 20-40>,
  "speed": <number 30-50>
}

No explanation, just the JSON.
"""
    
    response = model_v4.generate_content([prompt_v4, test_image])
    result_text = response.text.strip()
    
    # Try to extract JSON
    if '{' in result_text:
        json_start = result_text.index('{')
        json_end = result_text.rindex('}') + 1
        json_str = result_text[json_start:json_end]
        result = json.loads(json_str)
        
        print("\nResponse:")
        print(json.dumps(result, indent=2))
        
        if 'angle_degrees' in result:
            print(f"\n✅ SUCCESS! Got angle_degrees: {result['angle_degrees']}")
        else:
            print(f"\n❌ FAILED! No angle_degrees")
    else:
        print(f"\n❌ No JSON found in response:\n{result_text}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Run this script to quickly test different approaches.")
print("Whichever test succeeds, we'll implement that in gemini_bridge.py")
