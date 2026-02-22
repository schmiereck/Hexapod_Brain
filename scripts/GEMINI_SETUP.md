# Gemini API Setup Instructions

## Overview
Phase 3 requires Google Gemini API access for LLM-based robot control. The Gemini API will run on **ubuntu1** (Compute Node), not on the development PC.

## Prerequisites
- Google Cloud account with Gemini API access
- API key from https://makersuite.google.com/app/apikey
- SSH access to ubuntu1 (192.168.2.133)

## Setup Steps

### 1. SSH to ubuntu1
```bash
ssh ubuntu@192.168.2.133
cd ~/Hexapod_Brain
```

### 2. Run setup script
```bash
bash scripts/setup_gemini_ubuntu1.sh
```

This will:
- Check Python version (should be 3.10+)
- Install `google-generativeai` and `pillow`
- Verify imports

### 3. Configure API Key
Add to `~/.bashrc` on ubuntu1:
```bash
export GEMINI_API_KEY='your_api_key_here'
```

Then reload:
```bash
source ~/.bashrc
```

### 4. Test API Connection
```bash
python3 scripts/test_gemini_api.py
```

Expected output:
```
1. Testing import...
   ✅ google-generativeai imported successfully

2. Testing API key...
   ✅ API key found (length: XX)

3. Testing API connection...
   Available models:
     - models/gemini-2.0-flash-exp
     - models/gemini-1.5-pro
     ...
   ✅ API connection successful

4. Testing simple generation...
   Response: Hello Robot!
   ✅ Text generation successful

5. Testing JSON schema...
   Response: {"action": "move_forward", "confidence": 0.9}
   ✅ JSON schema enforcement successful

6. Testing multimodal input...
   Response: The image is red.
   ✅ Multimodal input successful

✅ All tests completed successfully!
```

## Troubleshooting

### Import Error
```
ModuleNotFoundError: No module named 'google.generativeai'
```
**Solution**: Run `pip3 install google-generativeai`

### API Key Error
```
❌ GEMINI_API_KEY not set
```
**Solution**: Export API key in `~/.bashrc` and `source ~/.bashrc`

### Connection Error
```
google.api_core.exceptions.PermissionDenied: 403 API key not valid
```
**Solution**: Check API key is correct at https://makersuite.google.com/app/apikey

### Model Not Found
```
google.api_core.exceptions.NotFound: 404 Model not found
```
**Solution**: Try different model name (e.g., `gemini-1.5-pro` instead of `gemini-2.0-flash-exp`)

## Next Steps
After successful API test:
1. Implement `gemini_bridge.py` node
2. Test with simulated inputs
3. Test with live robot and camera

## Cost Considerations
- Gemini 2.0 Flash: ~$0.01 per request (multimodal)
- Development: ~100 requests = $1
- Keep track at: https://console.cloud.google.com/billing

## Security
- **Never commit API key to git**
- Use environment variables only
- Rotate key if accidentally exposed
