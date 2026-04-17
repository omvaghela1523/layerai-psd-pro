from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import io, os, struct
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

app = Flask(__name__)
CORS(app)

REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")

# ── Struct helpers ────────────────────────────────────────────────
def pk(fmt, *a):
    return struct.pack(fmt, *a)

def pstring(s, pad=4):
    b = s.encode('ascii', errors='replace')[:255]
    raw = bytes([len(b)]) + b
    r = len(raw) % pad
    if r: raw += b'\x00' * (pad - r)
    return raw

def make_vignette(w, h):
    img = Image.new('RGBA', (w, h), (0,0,0,0))
    d = ImageDraw.Draw(img)
    for i in range(30):
        t = i / 30
        a = int(150*(1-t))
        x0,y0 = int(w*t*0.4), int(h*t*0.4)
        d.rectangle([x0,y0,w-x0,h-y0], outline=(0,0,0,a))
    return img.filter(ImageFilter.GaussianBlur(15))

# ── Additional Layer Information blocks ───────────────────────────

def make_brit_block(brightness=0, contrast=0):
    """
    'brit' — Brightness/Contrast adjustment layer descriptor
    PSD format: 4 bytes signature + 4 bytes key + 4 bytes length + data
    brit data: int16 brightness, int16 contrast, int16 mean, uint8 lab_only
    """
    data  = pk('>h', brightness)   # brightness -150..+150
    data += pk('>h', contrast)     # contrast -50..+100
    data += pk('>h', 128)          # mean value
    data += pk('>B', 0)            # lab color only: false
    return make_additional_info(b'brit', data)

def make_hue2_block(hue=0, saturation=0, lightness=0):
    """
    'hue2' — Hue/Saturation adjustment layer
    Version 2: uint16 version=2, then master + 6 ranges
    Each range: 6 int16 values
    """
    data = pk('>H', 2)  # version 2
    # Colorization flag
    data += pk('>B', 0)  # 0 = no colorization
    data += pk('>B', 0)  # padding
    # Master settings
    data += pk('>h', hue)          # master hue -180..+180
    data += pk('>h', saturation)   # master saturation -100..+100
    data += pk('>h', lightness)    # master lightness -100..+100
    # 6 color ranges (Reds, Yellows, Greens, Cyans, Blues, Magentas)
    for _ in range(6):
        # range begin, range end, range begin falloff, range end falloff
        data += pk('>hhhh', 0, 0, 0, 0)
        # hue, saturation, lightness for this range
        data += pk('>hhh', 0, 0, 0)
    return make_additional_info(b'hue2', data)

def make_levl_block(shadows=0, midtones=128, highlights=255, output_shadows=0, output_highlights=255):
    """
    'levl' — Levels adjustment layer
    Version 2 with composite + per-channel data
    """
    data = pk('>H', 2)  # version
    # Composite channel
    data += pk('>HHHHH', shadows, midtones, highlights, output_shadows, output_highlights)
    # RGB channels (R, G, B) — default values
    for _ in range(3):
        data += pk('>HHHHH', 0, 128, 255, 0, 255)
    return make_additional_info(b'levl', data)

def make_curv_block():
    """
    'curv' — Curves adjustment layer
    Simple default curves (identity curve)
    """
    data = pk('>H', 1)  # version
    data += pk('>I', 0)  # extra data count
    # Composite curve: 2 points (identity)
    data += pk('>H', 2)  # point count
    data += pk('>HH', 0, 0)      # point 1: output, input
    data += pk('>HH', 255, 255)  # point 2: output, input
    # 3 channel curves (R, G, B) — identity
    for _ in range(3):
        data += pk('>H', 2)
        data += pk('>HH', 0, 0)
        data += pk('>HH', 255, 255)
    return make_additional_info(b'curv', data)

def make_blnc_block(cyan_red=0, magenta_green=0, yellow_blue=0):
    """
    'blnc' — Color Balance adjustment layer
    3 sets of 3 int16 values (shadows, midtones, highlights)
    + uint8 preserve luminosity
    """
    data = b''
    # Shadows
    data += pk('>hhh', 0, 0, 0)
    # Midtones
    data += pk('>hhh', cyan_red, magenta_green, yellow_blue)
    # Highlights
    data += pk('>hhh', 0, 0, 0)
    # Preserve luminosity
    data += pk('>B', 1)
    return make_additional_info(b'blnc', data)

def make_luni_block(name):
    """
    'luni' — Unicode layer name
    """
    encoded = name.encode('utf-16-be')
    data  = pk('>I', len(name))  # character count
    data += encoded
    return make_additional_info(b'luni', data)

def make_lsct_block(section_type=0):
    """
    'lsct' — Section divider (for groups)
    0=other, 1=open folder, 2=closed folder, 3=bounding section divider
    """
    data = pk('>I', section_type)
    return make_additional_info(b'lsct', data)

def make_additional_info(key, data):
    """Wrap data in 8BIM additional layer info block"""
    block  = b'8BIM'
    block += key
    block += pk('>I', len(data))
    block += data
    # Pad to even
    if len(block) % 2:
        block += b'\x00'
    return block

# ── PSD builder with real adjustment layers ───────────────────────
def build_psd(layers, W, H):
    """
    Build PSD with proper adjustment layers that are editable in Photoshop/Photopea.
    
    Each adjustment layer is a zero-pixel layer with the correct
    Additional Layer Information block that Photoshop recognizes.
    """

    # ── Section 1: Header ─────────────────────────────────────────
    s1  = b'8BPS'
    s1 += pk('>H', 1)
    s1 += b'\x00' * 6
    s1 += pk('>H', 4)   # channels
    s1 += pk('>I', H)
    s1 += pk('>I', W)
    s1 += pk('>H', 8)   # depth
    s1 += pk('>H', 3)   # RGB

    s2 = pk('>I', 0)     # color mode
    s3 = pk('>I', 0)     # image resources

    # ── Build layers ──────────────────────────────────────────────
    records = b''
    chandata = b''

    for layer in layers:
        ltype = layer.get('type', 'pixel')
        name  = layer['name']
        blend = layer.get('blend_mode', 'norm').encode('ascii').ljust(4)[:4]
        opac  = layer.get('opacity', 255)

        if ltype == 'pixel':
            # Normal pixel layer
            img = layer['image'].convert('RGBA').resize((W, H), Image.LANCZOS)
            arr = np.array(img, dtype=np.uint8)

            chs = [(-1,3),(0,0),(1,1),(2,2)]
            parts = []
            for cid, ci in chs:
                cb = pk('>H', 0) + arr[:,:,ci].tobytes()
                parts.append((cid, cb))

            rec  = pk('>IIII', 0, 0, H, W)
            rec += pk('>H', 4)
            for cid, cb in parts:
                rec += pk('>hI', cid, len(cb))
            rec += b'8BIM' + blend
            rec += pk('>BBBB', opac, 0, 0, 0)

            extra  = pk('>I', 0)  # mask
            extra += pk('>I', 0)  # blend ranges
            extra += pstring(name, 4)
            extra += make_luni_block(name)

            rec += pk('>I', len(extra)) + extra
            records += rec
            for _, cb in parts:
                chandata += cb

        elif ltype == 'adjustment':
            # Adjustment layer — zero-size pixel data, just the adj block
            # Need at least a transparency channel
            # Create a white full-opacity mask
            mask_data = np.full((H, W), 255, dtype=np.uint8)
            mask_raw = pk('>H', 0) + mask_data.tobytes()

            rec  = pk('>IIII', 0, 0, H, W)
            rec += pk('>H', 1)  # just 1 channel: user mask
            rec += pk('>hI', -1, len(mask_raw))

            rec += b'8BIM' + blend
            rec += pk('>BBBB', opac, 0, 0, 0)

            # Extra data with adjustment info
            extra  = pk('>I', 0)   # layer mask
            extra += pk('>I', 0)   # blend ranges
            extra += pstring(name, 4)
            extra += make_luni_block(name)

            # Add the specific adjustment layer block
            adj_key = layer.get('adj_key', '')
            if adj_key == 'brit':
                extra += make_brit_block(
                    layer.get('brightness', 0),
                    layer.get('contrast', 0))
            elif adj_key == 'hue2':
                extra += make_hue2_block(
                    layer.get('hue', 0),
                    layer.get('saturation', 0),
                    layer.get('lightness', 0))
            elif adj_key == 'curv':
                extra += make_curv_block()
            elif adj_key == 'levl':
                extra += make_levl_block()
            elif adj_key == 'blnc':
                extra += make_blnc_block(
                    layer.get('cyan_red', 0),
                    layer.get('magenta_green', 0),
                    layer.get('yellow_blue', 0))

            rec += pk('>I', len(extra)) + extra
            records += rec
            chandata += mask_raw

    # Layer info
    li = pk('>h', len(layers)) + records + chandata
    if len(li) % 2: li += b'\x00'

    body = pk('>I', len(li)) + li + pk('>I', 0)
    s4 = pk('>I', len(body)) + body

    # ── Merged composite ──────────────────────────────────────────
    merged = Image.new('RGBA', (W, H), (0,0,0,255))
    for layer in reversed(layers):
        if layer.get('type') == 'pixel' and 'image' in layer:
            limg = layer['image'].convert('RGBA').resize((W,H), Image.LANCZOS)
            merged = Image.alpha_composite(merged, limg)

    rgb = np.array(merged.convert('RGB'), dtype=np.uint8)
    s5  = pk('>H', 0)
    s5 += rgb[:,:,0].tobytes()
    s5 += rgb[:,:,1].tobytes()
    s5 += rgb[:,:,2].tobytes()

    return s1 + s2 + s3 + s4 + s5

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "LayerAI PSD Pro", "version": "9.0.0"})

@app.route('/generate-psd', methods=['POST'])
def generate_psd():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        raw   = request.files['image'].read()
        orig  = Image.open(io.BytesIO(raw)).convert('RGBA')
        W, H  = orig.size

        MAX = 1000
        if W > MAX or H > MAX:
            r = min(MAX/W, MAX/H)
            W, H = int(W*r), int(H*r)
            orig = orig.resize((W,H), Image.LANCZOS)

        layers = []

        # 1. Background (pixel)
        layers.append({
            'type': 'pixel',
            'name': 'Background',
            'image': orig.copy(),
            'blend_mode': 'norm',
            'opacity': 255,
        })

        # 2. Subject Masked (pixel) — Remove.bg
        if REMOVE_BG_API_KEY:
            try:
                rsp = requests.post(
                    'https://api.remove.bg/v1.0/removebg',
                    files={'image_file': ('i.jpg', raw, 'image/jpeg')},
                    data={'size': 'auto'},
                    headers={'X-Api-Key': REMOVE_BG_API_KEY},
                    timeout=20)
                if rsp.status_code == 200:
                    subj = Image.open(io.BytesIO(rsp.content)).convert('RGBA')
                    subj = subj.resize((W, H), Image.LANCZOS)
                    layers.append({
                        'type': 'pixel',
                        'name': 'Subject — Masked',
                        'image': subj,
                        'blend_mode': 'norm',
                        'opacity': 255,
                    })
            except Exception as e:
                print('removebg:', e)

        # 3. Curves (adjustment)
        layers.append({
            'type': 'adjustment',
            'name': 'Curves 1',
            'adj_key': 'curv',
            'blend_mode': 'norm',
            'opacity': 255,
        })

        # 4. Brightness/Contrast (adjustment)
        layers.append({
            'type': 'adjustment',
            'name': 'Brightness/Contrast 1',
            'adj_key': 'brit',
            'brightness': 15,
            'contrast': 20,
            'blend_mode': 'norm',
            'opacity': 255,
        })

        # 5. Hue/Saturation (adjustment)
        layers.append({
            'type': 'adjustment',
            'name': 'Hue/Saturation 1',
            'adj_key': 'hue2',
            'hue': 0,
            'saturation': 15,
            'lightness': 5,
            'blend_mode': 'norm',
            'opacity': 255,
        })

        # 6. Color Balance (adjustment)
        layers.append({
            'type': 'adjustment',
            'name': 'Color Balance 1',
            'adj_key': 'blnc',
            'cyan_red': -10,
            'magenta_green': 5,
            'yellow_blue': 15,
            'blend_mode': 'norm',
            'opacity': 255,
        })

        # 7. Levels (adjustment)
        layers.append({
            'type': 'adjustment',
            'name': 'Levels 1',
            'adj_key': 'levl',
            'blend_mode': 'norm',
            'opacity': 255,
        })

        # 8. Vignette (pixel)
        layers.append({
            'type': 'pixel',
            'name': 'Vignette',
            'image': make_vignette(W, H),
            'blend_mode': 'mul ',
            'opacity': 180,
        })

        psd = build_psd(layers, W, H)
        buf = io.BytesIO(psd)
        buf.seek(0)

        return send_file(buf, mimetype='application/octet-stream',
                         as_attachment=True, download_name='layerai-export.psd')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
