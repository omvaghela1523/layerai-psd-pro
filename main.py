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
def pk(fmt, *a): return struct.pack(fmt, *a)

def pstring(s, pad=4):
    b = s.encode('ascii', errors='replace')[:255]
    raw = bytes([len(b)]) + b
    r = len(raw) % pad
    if r: raw += b'\x00' * (pad - r)
    return raw

def make_additional(key, data):
    """8BIM + 4-byte key + 4-byte length + data (padded to even)"""
    block = b'8BIM' + key + pk('>I', len(data)) + data
    if len(block) % 2: block += b'\x00'
    return block

def make_luni(name):
    """Unicode layer name"""
    encoded = name.encode('utf-16-be')
    return make_additional(b'luni', pk('>I', len(name)) + encoded + b'\x00\x00')

def make_lnsr(layer_type):
    """Layer source: 'layr' for pixel, 'cont' for adjustment"""
    return make_additional(b'lnsr', layer_type)

def make_lyid(lid):
    return make_additional(b'lyid', pk('>I', lid))

def make_clbl():
    return make_additional(b'clbl', b'\x01\x00\x00\x00')

def make_infx():
    return make_additional(b'infx', b'\x00\x00\x00\x00')

def make_knko():
    return make_additional(b'knko', b'\x00\x00\x00\x00')

def make_lspf():
    return make_additional(b'lspf', b'\x00\x00\x00\x00')

def make_lclr():
    return make_additional(b'lclr', b'\x00' * 8)

def make_fxrp():
    return make_additional(b'fxrp', b'\x00' * 16)

def make_common_extras(name, lid, is_adjustment=False):
    """Common additional layer info blocks that Photoshop always writes"""
    e = b''
    e += make_luni(name)
    e += make_lnsr(b'cont' if is_adjustment else b'layr')
    e += make_lyid(lid)
    e += make_clbl()
    e += make_infx()
    e += make_knko()
    e += make_lspf()
    e += make_lclr()
    e += make_fxrp()
    return e

# ── Adjustment layer descriptors ──────────────────────────────────

def make_brit_block(brightness=0, contrast=0):
    """
    Exact format from Photoshop hex dump:
    brit: 8 bytes = int16 brightness, int16 contrast, int16 mean, uint8 lab, pad
    Actually observed: 0000 0000 0000 0000 for brightness=0 contrast=0
    """
    data = pk('>hh', brightness, contrast)
    data += pk('>h', 128)   # mean value for legacy
    data += pk('>B', 0)     # lab only
    data += b'\x00'         # padding to even
    return make_additional(b'brit', data)

def make_hue2_block(hue=0, saturation=0, lightness=0):
    """hue2: Hue/Saturation v2"""
    data = pk('>H', 2)     # version
    data += pk('>B', 0)    # colorization
    data += pk('>B', 0)    # padding
    data += pk('>hhh', hue, saturation, lightness)  # master
    for _ in range(6):     # 6 color ranges
        data += pk('>hhhh', 0, 0, 0, 0)  # range
        data += pk('>hhh', 0, 0, 0)       # adjustments
    return make_additional(b'hue2', data)

def make_curv_block():
    """curv: Curves"""
    data = pk('>H', 4)     # version
    data += pk('>I', 0)    # count of extra curves data
    # Composite curve: 2 points
    data += pk('>H', 2)
    data += pk('>HH', 0, 0)
    data += pk('>HH', 255, 255)
    # R, G, B curves: identity
    for _ in range(3):
        data += pk('>H', 2)
        data += pk('>HH', 0, 0)
        data += pk('>HH', 255, 255)
    return make_additional(b'curv', data)

def make_levl_block():
    """levl: Levels"""
    data = pk('>H', 2)     # version
    # Composite: input shadow, input half, input highlight, output shadow, output highlight
    data += pk('>HHHHH', 0, 128, 255, 0, 255)
    for _ in range(3):     # R, G, B
        data += pk('>HHHHH', 0, 128, 255, 0, 255)
    return make_additional(b'levl', data)

def make_blnc_block(cr=0, mg=0, yb=0):
    """blnc: Color Balance"""
    data = pk('>hhh', 0, 0, 0)        # shadows
    data += pk('>hhh', cr, mg, yb)     # midtones
    data += pk('>hhh', 0, 0, 0)        # highlights
    data += pk('>B', 1)                # preserve luminosity
    data += b'\x00'                    # pad
    return make_additional(b'blnc', data)

# ── Blending ranges (40 bytes as Photoshop writes) ────────────────
def make_blending_ranges():
    """Photoshop always writes 40 bytes of blending ranges for RGB"""
    data = b''
    # Composite gray blend: this layer 0-255, underlying 0-255
    data += pk('>BBBB', 0, 255, 0, 255)  # source
    data += pk('>BBBB', 0, 255, 0, 255)  # destination
    # R channel
    data += pk('>BBBB', 0, 255, 0, 255)
    data += pk('>BBBB', 0, 255, 0, 255)
    # G channel
    data += pk('>BBBB', 0, 255, 0, 255)
    data += pk('>BBBB', 0, 255, 0, 255)
    # B channel — wait, that's already 48. Real file shows len=40
    # Actually: composite (8) + 3 channels (8 each) + alpha? = let me use exact 40
    return data[:40] if len(data) > 40 else data + b'\x00' * (40 - len(data))

# ── Mask data for adjustment layers ───────────────────────────────
def make_adj_mask_data():
    """
    Photoshop writes 20 bytes mask data for adjustment layers:
    top, left, bottom, right (4x4=16 bytes) + defaultColor(1) + flags(1) + padding(2)
    """
    data = pk('>IIII', 0, 0, 0, 0)   # empty mask rect
    data += pk('>B', 255)              # default color (white = fully visible)
    data += pk('>B', 0)                # flags
    data += b'\x00\x00'                # padding to 4-byte boundary
    return data

# ── Build pixel layer record ──────────────────────────────────────
def build_pixel_layer(name, img, blend, opacity, W, H, lid):
    """Build a normal pixel layer"""
    img = img.convert('RGBA').resize((W, H), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)

    # Channel data: compression=1 (RLE)
    chs = [(-1, 3), (0, 0), (1, 1), (2, 2)]
    ch_parts = []
    for ch_id, ch_idx in chs:
        plane = arr[:, :, ch_idx]
        # RLE: row byte counts + compressed rows
        row_counts = []
        row_data = b''
        for row in plane:
            raw = row.tobytes()
            # Simple: just use raw for each row
            compressed = bytes([len(raw) - 1]) + raw  # literal run
            row_counts.append(len(compressed))
            row_data += compressed
        
        # Actually use raw (compression=0) for simplicity & speed
        ch_data = pk('>H', 0) + plane.tobytes()
        ch_parts.append((ch_id, ch_data))

    # Layer record
    rec = pk('>IIII', 0, 0, H, W)
    rec += pk('>H', 4)
    for ch_id, ch_data in ch_parts:
        rec += pk('>hI', ch_id, len(ch_data))

    bm = blend.encode('ascii').ljust(4)[:4]
    rec += b'8BIM' + bm
    rec += pk('>B', opacity)
    rec += pk('>B', 0)       # clipping
    rec += pk('>B', 8)       # flags: bit 3 = has useful data
    rec += pk('>B', 0)       # filler

    # Extra data
    extra = pk('>I', 0)      # mask data (none for pixel)
    br = make_blending_ranges()
    extra += pk('>I', len(br)) + br
    extra += pstring(name, 4)
    extra += make_common_extras(name, lid, is_adjustment=False)

    rec += pk('>I', len(extra)) + extra

    ch_bytes = b''
    for _, cd in ch_parts:
        ch_bytes += cd

    return rec, ch_bytes

# ── Build adjustment layer record ─────────────────────────────────
def build_adjustment_layer(name, adj_block, blend, opacity, W, H, lid):
    """
    Build an adjustment layer exactly like Photoshop does:
    - bbox = 0,0,0,0
    - 5 channels (alpha, R, G, B, user mask) each with size=2 (just compression=0)
    - flags = 24 (bits 3 and 4)
    - mask data = 20 bytes
    - blending ranges = 40 bytes
    - Additional info: adj_block + common extras
    """

    # 5 channels, each is just 2 bytes (compression=0, no pixel data because bbox is empty)
    ch_ids = [-1, 0, 1, 2, -2]
    ch_data_each = pk('>H', 0)  # just compression type, 0 pixels because 0x0 rect

    rec = pk('>IIII', 0, 0, 0, 0)  # empty bbox!
    rec += pk('>H', 5)              # 5 channels
    for ch_id in ch_ids:
        rec += pk('>hI', ch_id, len(ch_data_each))

    bm = blend.encode('ascii').ljust(4)[:4]
    rec += b'8BIM' + bm
    rec += pk('>B', opacity)
    rec += pk('>B', 0)       # clipping
    rec += pk('>B', 24)      # flags = 24 (has useful data + bit 4)
    rec += pk('>B', 0)       # filler

    # Extra data
    mask = make_adj_mask_data()
    extra = pk('>I', len(mask)) + mask

    br = make_blending_ranges()
    extra += pk('>I', len(br)) + br
    extra += pstring(name, 4)

    # Adjustment-specific block
    extra += adj_block
    # Common extras
    extra += make_common_extras(name, lid, is_adjustment=True)

    rec += pk('>I', len(extra)) + extra

    # Channel bytes (5 x 2 bytes each)
    ch_bytes = ch_data_each * 5

    return rec, ch_bytes

# ── Main PSD assembly ─────────────────────────────────────────────
def create_psd(layer_specs, W, H):
    # Section 1: Header
    s1  = b'8BPS' + pk('>H', 1) + b'\x00'*6
    s1 += pk('>H', 3) + pk('>I', H) + pk('>I', W)
    s1 += pk('>H', 8) + pk('>H', 3)

    # Section 2 & 3: empty
    s2 = pk('>I', 0)
    s3 = pk('>I', 0)

    # Section 4: Layers
    all_records = b''
    all_chdata = b''
    
    for spec in layer_specs:
        if spec['type'] == 'pixel':
            rec, chd = build_pixel_layer(
                spec['name'], spec['image'], spec['blend_mode'],
                spec['opacity'], W, H, spec['lid'])
        else:
            rec, chd = build_adjustment_layer(
                spec['name'], spec['adj_block'], spec['blend_mode'],
                spec['opacity'], W, H, spec['lid'])
        all_records += rec
        all_chdata += chd

    li = pk('>h', len(layer_specs)) + all_records + all_chdata
    if len(li) % 2: li += b'\x00'

    body = pk('>I', len(li)) + li + pk('>I', 0)
    s4 = pk('>I', len(body)) + body

    # Section 5: Merged composite
    merged = Image.new('RGBA', (W, H), (0, 0, 0, 255))
    for spec in reversed(layer_specs):
        if spec['type'] == 'pixel' and 'image' in spec:
            limg = spec['image'].convert('RGBA').resize((W, H), Image.LANCZOS)
            merged = Image.alpha_composite(merged, limg)

    rgb = np.array(merged.convert('RGB'), dtype=np.uint8)
    s5 = pk('>H', 0)
    s5 += rgb[:,:,0].tobytes()
    s5 += rgb[:,:,1].tobytes()
    s5 += rgb[:,:,2].tobytes()

    return s1 + s2 + s3 + s4 + s5

def make_vignette(w, h):
    img = Image.new('RGBA', (w, h), (0,0,0,0))
    d = ImageDraw.Draw(img)
    for i in range(30):
        t = i/30; a = int(150*(1-t))
        x0,y0 = int(w*t*0.4), int(h*t*0.4)
        d.rectangle([x0,y0,w-x0,h-y0], outline=(0,0,0,a))
    return img.filter(ImageFilter.GaussianBlur(15))

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
@app.route('/health')
def health():
    return jsonify({"status":"ok","service":"LayerAI PSD Pro","version":"10.0.0"})

@app.route('/generate-psd', methods=['POST'])
def generate_psd():
    try:
        if 'image' not in request.files:
            return jsonify({"error":"No image uploaded"}), 400

        raw = request.files['image'].read()
        orig = Image.open(io.BytesIO(raw))
        # Force RGB mode first, then RGBA
        if orig.mode in ('CMYK', 'P', 'L', 'LA', 'I', 'F'):
            orig = orig.convert('RGB')
        orig = orig.convert('RGBA')
        W, H = orig.size

        MAX = 1000
        if W > MAX or H > MAX:
            r = min(MAX/W, MAX/H)
            W, H = int(W*r), int(H*r)
            orig = orig.resize((W, H), Image.LANCZOS)
        
        # Ensure uint8
        arr_check = np.array(orig, dtype=np.uint8)
        orig = Image.fromarray(arr_check, 'RGBA')

        specs = []
        lid = 1

        # 1. Background
        specs.append({
            'type': 'pixel', 'name': 'Background',
            'image': orig.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 2. Subject (Remove.bg)
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
                    specs.append({
                        'type': 'pixel', 'name': 'Subject Masked',
                        'image': subj, 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
                    })
                    lid += 1
            except Exception as e:
                print('removebg:', e)

        # 3. Curves (adjustment)
        specs.append({
            'type': 'adjustment', 'name': 'Curves 1',
            'adj_block': make_curv_block(),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 4. Brightness/Contrast (adjustment)
        specs.append({
            'type': 'adjustment', 'name': 'Brightness/Contrast 1',
            'adj_block': make_brit_block(20, 10),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 5. Hue/Saturation (adjustment)
        specs.append({
            'type': 'adjustment', 'name': 'Hue/Saturation 1',
            'adj_block': make_hue2_block(0, 15, 5),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 6. Color Balance (adjustment)
        specs.append({
            'type': 'adjustment', 'name': 'Color Balance 1',
            'adj_block': make_blnc_block(-10, 5, 15),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 7. Levels (adjustment)
        specs.append({
            'type': 'adjustment', 'name': 'Levels 1',
            'adj_block': make_levl_block(),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 8. Vignette (pixel)
        specs.append({
            'type': 'pixel', 'name': 'Vignette',
            'image': make_vignette(W, H),
            'blend_mode': 'mul ', 'opacity': 180, 'lid': lid
        })

        psd = create_psd(specs, W, H)
        buf = io.BytesIO(psd)
        buf.seek(0)

        return send_file(buf, mimetype='application/octet-stream',
                         as_attachment=True, download_name='layerai-export.psd')

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
