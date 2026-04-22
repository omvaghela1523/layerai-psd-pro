# LayerAI V2 — Complete Setup Guide
# ====================================

## Tumhare paas kya hai:

```
app.py              ← V2 fixed Flask app (YEH REPLACE KARO)
requirements.txt    ← Python dependencies
server.js           ← AI analysis (Railway pe, change mat karo)
```

## Step-by-Step Setup:

### STEP 1: GitHub pe code update karo

Apne GitHub repo mein jao aur yeh 2 files replace karo:

1. **app.py** → Purani `app.py` delete karo, naye se replace karo
2. **requirements.txt** → Replace karo (ya verify karo ki sab packages hain)

```bash
# Terminal mein (agar git use karte ho):
cd your-layerai-repo
# Naye files copy karo repo mein
git add app.py requirements.txt
git commit -m "V2: Fix hue2, levl, blnc adjustment layers + Vision debug"
git push origin main
```

### STEP 2: Render pe deploy karo

1. Render dashboard jao: https://dashboard.render.com
2. Apna LayerAI service select karo
3. Agar GitHub auto-deploy ON hai → Push karte hi deploy ho jayega
4. Agar manual hai → "Manual Deploy" → "Deploy latest commit" click karo

**Render Settings verify karo:**
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
- Environment Variables:
  - REMOVE_BG_API_KEY = tumhari key
  - GOOGLE_VISION_API_KEY = tumhari key

### STEP 3: Test karo

Deploy hone ke baad (2-3 minute lagenge), yeh URLs try karo:

**Health check:**
```
GET https://layerai-psd-pro.onrender.com/health
```
Response mein dikhega:
```json
{
  "status": "ok",
  "version": "2.0.0",
  "features": {
    "brit": "working (legacy)",
    "curv": "working (legacy)",
    "hue2": "FIXED (descriptor)",
    "levl": "FIXED (descriptor)",
    "blnc": "FIXED (descriptor)",
    "vision": "enabled",
    "removebg": "enabled"
  }
}
```

**Environment check:**
```
GET https://layerai-psd-pro.onrender.com/debug-env
```

**Test all V2 adjustment layers (IMPORTANT!):**
```
GET https://layerai-psd-pro.onrender.com/test-adjustments
```
→ Yeh PSD download hoga
→ Isko Photoshop 2026 mein kholo
→ Verify karo:
  ✓ File bina error ke khule
  ✓ Layers panel mein 6 layers dikhein
  ✓ Hue/Saturation pe double-click karo → dialog khule with values
  ✓ Levels pe double-click karo → dialog khule with values
  ✓ Color Balance pe double-click karo → dialog khule with values
  ✓ Brightness/Contrast pe double-click karo → dialog khule with values

**Test individual layers:**
```
GET https://layerai-psd-pro.onrender.com/test-single-layer?type=hue2
GET https://layerai-psd-pro.onrender.com/test-single-layer?type=levl
GET https://layerai-psd-pro.onrender.com/test-single-layer?type=blnc
GET https://layerai-psd-pro.onrender.com/test-single-layer?type=brit
GET https://layerai-psd-pro.onrender.com/test-single-layer?type=curv
```

**Test Google Vision (Week 2 debug):**
```bash
curl -X POST https://layerai-psd-pro.onrender.com/test-vision \
  -F "image=@test-photo-with-text.jpg"
```
→ Response mein dikhega ki text detect hua ya nahi, aur kya error aaya

**Test dynamic PSD with all layers:**
```bash
curl -X POST https://layerai-psd-pro.onrender.com/generate-psd-dynamic \
  -F "image=@your-photo.jpg" \
  -F "brightness=15" \
  -F "contrast=10" \
  -F "hue=5" \
  -F "saturation=-10" \
  -F "lightness=0" \
  -F "lvl_shadows=10" \
  -F "lvl_midtones=1.15" \
  -F "lvl_highlights=245" \
  -F "cb_midtone_cr=8" \
  -F "cb_midtone_yb=-10" \
  -F "color_grade=Cinematic" \
  --output test-dynamic.psd
```

### STEP 4: Verify PSD structure (optional debug)

Agar koi PSD corrupt lage toh:
```bash
curl -X POST https://layerai-psd-pro.onrender.com/verify-psd \
  -F "file=@your-file.psd"
```
→ Yeh batayega ki kaun kaun se blocks hain aur kis offset pe

---

## V1 vs V2 Comparison:

| Feature              | V1 (Broken)          | V2 (Fixed)              |
|----------------------|----------------------|-------------------------|
| brit block           | ✅ Working (legacy)  | ✅ Working (legacy)     |
| curv block           | ✅ Working (legacy)  | ✅ Working (legacy)     |
| hue2 block           | ❌ Corrupts PSD      | ✅ Descriptor format    |
| levl block           | ❌ Corrupts PSD      | ✅ Descriptor format    |
| blnc block           | ❌ Corrupts PSD      | ✅ Descriptor format    |
| Vision text debug    | ❌ Silent fail       | ✅ /test-vision endpoint|
| Dynamic all layers   | ❌ Missing hue/lvl   | ✅ All params accepted  |
| Test endpoints       | ❌ None              | ✅ 5 test endpoints     |

## Kya badla V2 mein:

1. **hue2 block**: Raw uint16 format → Descriptor format (version 16, HStr class)
2. **levl block**: Raw uint16 format → Descriptor format (version 16, Lvls class)
3. **blnc block**: Raw int16 format → Descriptor format (version 16, ClrB class)
4. **detect_text()**: Added detailed logging at every step
5. **New endpoints**: /test-vision, /test-adjustments, /test-single-layer, /verify-psd, /debug-env
6. **Dynamic PSD**: Now accepts hue, saturation, lightness, levels, color balance params

## Architecture:

```
User (Emergent.sh UI)
  │
  ├──→ Railway (Node.js) ── /analyze-image ── Claude AI analysis
  │         │
  │         ▼ returns JSON with adjustment values
  │
  └──→ Render (Python Flask) ── /generate-psd-dynamic
            │
            ├── Background pixel layer
            ├── Subject Masked (Remove.bg)
            ├── Curves adjustment (editable)
            ├── Brightness/Contrast adjustment (editable)
            ├── Hue/Saturation adjustment (V2 FIX, editable)
            ├── Levels adjustment (V2 FIX, editable)
            ├── Color Balance adjustment (V2 FIX, editable)
            ├── Vignette pixel layer
            ├── Color Grade pixel layer
            └── Text layers (Google Vision)
            │
            ▼
        layerai-export.psd (Photoshop 2026 compatible)
```
