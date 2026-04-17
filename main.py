from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import io, os, struct
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

app = Flask(__name__)
CORS(app)

REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")

def pk(fmt, *a): return struct.pack(fmt, *a)

def pstring(s, pad=4):
    b = s.encode('ascii', errors='replace')[:255]
    raw = bytes([len(b)]) + b
    r = len(raw) % pad
    if r: raw += b'\x00' * (pad - r)
    return raw

def make_additional(key, data):
    block = b'8BIM' + key + pk('>I', len(data)) + data
    if len(block) % 2: block += b'\x00'
    return block

def make_luni(name):
    encoded = name.encode('utf-16-be')
    return make_additional(b'luni', pk('>I', len(name)) + encoded + b'\x00\x00')

def make_lnsr(t): return make_additional(b'lnsr', t)
def make_lyid(lid): return make_additional(b'lyid', pk('>I', lid))
def make_clbl(): return make_additional(b'clbl', b'\x01\x00\x00\x00')
def make_infx(): return make_additional(b'infx', b'\x00\x00\x00\x00')
def make_knko(): return make_additional(b'knko', b'\x00\x00\x00\x00')
def make_lspf(): return make_additional(b'lspf', b'\x00\x00\x00\x00')
def make_lclr(): return make_additional(b'lclr', b'\x00' * 8)
def make_fxrp(): return make_additional(b'fxrp', b'\x00' * 16)

def make_common_extras(name, lid, is_adj=False):
    e = make_luni(name)
    e += make_lnsr(b'cont' if is_adj else b'layr')
    e += make_lyid(lid)
    e += make_clbl()
    e += make_infx()
    e += make_knko()
    e += make_lspf()
    e += make_lclr()
    e += make_fxrp()
    return e

def make_brit_block(brightness=0, contrast=0):
    data = pk('>hh', brightness, contrast) + pk('>h', 128) + pk('>B', 0) + b'\x00'
    return make_additional(b'brit', data)

def make_hue2_block(hue=0, saturation=0, lightness=0):
    data = pk('>H', 2) + pk('>BB', 0, 0) + pk('>hhh', hue, saturation, lightness)
    for _ in range(6):
        data += pk('>hhhh', 0, 0, 0, 0) + pk('>hhh', 0, 0, 0)
    return make_additional(b'hue2', data)

def make_curv_block():
    data = pk('>H', 4) + pk('>I', 0)
    data += pk('>H', 2) + pk('>HH', 0, 0) + pk('>HH', 255, 255)
    for _ in range(3):
        data += pk('>H', 2) + pk('>HH', 0, 0) + pk('>HH', 255, 255)
    return make_additional(b'curv', data)

def make_levl_block():
    data = pk('>H', 2)
    data += pk('>HHHHH', 0, 128, 255, 0, 255)
    for _ in range(3):
        data += pk('>HHHHH', 0, 128, 255, 0, 255)
    return make_additional(b'levl', data)

def make_blnc_block(cr=0, mg=0, yb=0):
    data = pk('>hhh', 0, 0, 0) + pk('>hhh', cr, mg, yb) + pk('>hhh', 0, 0, 0)
    data += pk('>B', 1) + b'\x00'
    return make_additional(b'blnc', data)

def make_blending_ranges():
    data = b''
    data += pk('>BBBB', 0, 255, 0, 255) + pk('>BBBB', 0, 255, 0, 255)
    for _ in range(3):
        data += pk('>BBBB', 0, 255, 0, 255) + pk('>BBBB', 0, 255, 0, 255)
    return data

def make_adj_mask_data():
    return pk('>IIII', 0, 0, 0, 0) + pk('>BB', 255, 0) + b'\x00\x00'

def make_vignette(w, h):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(30):
        t = i / 30
        a = int(150 * (1 - t))
        x0, y0 = int(w * t * 0.4), int(h * t * 0.4)
        d.rectangle([x0, y0, w - x0, h - y0], outline=(0, 0, 0, a))
    return img.filter(ImageFilter.GaussianBlur(15))

# ── Build pixel layer ─────────────────────────────────────────────
def build_pixel_layer(name, img, blend, opacity, W, H, lid):
    img = img.convert('RGBA').resize((W, H), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)

    # 4 channels: Alpha(-1), R(0), G(1), B(2)
    chs = [(-1, 3), (0, 0), (1, 1), (2, 2)]
    ch_parts = []
    for ch_id, ch_idx in chs:
        if ch_id == -1 and name == 'Background':
            alpha_full = np.full((H, W), 255, dtype=np.uint8)
            ch_data = pk('>H', 0) + alpha_full.tobytes()
        else:
            ch_data = pk('>H', 0) + arr[:, :, ch_idx].tobytes()
        ch_parts.append((ch_id, ch_data))

    rec = pk('>IIII', 0, 0, H, W)
    rec += pk('>H', 4)
    for ch_id, ch_data in ch_parts:
        rec += pk('>hI', ch_id, len(ch_data))

    bm = blend.encode('ascii').ljust(4)[:4]
    rec += b'8BIM' + bm
    rec += pk('>BBBB', opacity, 0, 8, 0)

    extra = pk('>I', 0)
    br = make_blending_ranges()
    extra += pk('>I', len(br)) + br
    extra += pstring(name, 4)
    extra += make_common_extras(name, lid, is_adj=False)

    rec += pk('>I', len(extra)) + extra

    ch_bytes = b''.join(cd for _, cd in ch_parts)
    return rec, ch_bytes

# ── Build adjustment layer ────────────────────────────────────────
def build_adjustment_layer(name, adj_block, blend, opacity, W, H, lid):
    ch_ids = [-1, 0, 1, 2, -2]
    ch_data_each = pk('>H', 0)

    rec = pk('>IIII', 0, 0, 0, 0)
    rec += pk('>H', 5)
    for ch_id in ch_ids:
        rec += pk('>hI', ch_id, len(ch_data_each))

    bm = blend.encode('ascii').ljust(4)[:4]
    rec += b'8BIM' + bm
    rec += pk('>BBBB', opacity, 0, 24, 0)

    mask = make_adj_mask_data()
    extra = pk('>I', len(mask)) + mask
    br = make_blending_ranges()
    extra += pk('>I', len(br)) + br
    extra += pstring(name, 4)
    extra += adj_block
    extra += make_common_extras(name, lid, is_adj=True)

    rec += pk('>I', len(extra)) + extra
    ch_bytes = ch_data_each * 5
    return rec, ch_bytes

# ── Create PSD ────────────────────────────────────────────────────
def create_psd(layer_specs, W, H):
    # Header: 4 channels (RGBA) so merged composite has alpha too
    s1 = b'8BPS' + pk('>H', 1) + b'\x00' * 6
    s1 += pk('>H', 3) + pk('>I', H) + pk('>I', W)
    s1 += pk('>H', 8) + pk('>H', 3)

    s2 = pk('>I', 0)
    s3 = pk('>I', 0)

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

    # Use negative layer count to indicate merged alpha is present
    li = pk('>h', len(layer_specs)) + all_records + all_chdata
    if len(li) % 2: li += b'\x00'

    body = pk('>I', len(li)) + li + pk('>I', 0)
    s4 = pk('>I', len(body)) + body

    # Merged composite — flatten all visible pixel layers
    merged = Image.new('RGBA', (W, H), (255, 255, 255, 255))
    for spec in layer_specs:
        if spec['type'] == 'pixel' and 'image' in spec:
            limg = spec['image'].convert('RGBA').resize((W, H), Image.LANCZOS)
            merged = Image.alpha_composite(merged, limg)

    # 4 channels: A, R, G, B (because header says 4 channels)
    merged_rgb = np.array(merged.convert('RGB'), dtype=np.uint8)
    s5 = pk('>H', 0)
    s5 += merged_rgb[:, :, 0].tobytes()
    s5 += merged_rgb[:, :, 1].tobytes()
    s5 += merged_rgb[:, :, 2].tobytes()

    return s1 + s2 + s3 + s4 + s5

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "LayerAI PSD Pro", "version": "11.0.0"})

@app.route('/generate-psd', methods=['POST'])
def generate_psd():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        raw = request.files['image'].read()
        orig = Image.open(io.BytesIO(raw))
        if orig.mode in ('CMYK', 'P', 'L', 'LA', 'I', 'F'):
            orig = orig.convert('RGB')
        orig = orig.convert('RGBA')
        W, H = orig.size

        MAX = 1000
        if W > MAX or H > MAX:
            r = min(MAX / W, MAX / H)
            W, H = int(W * r), int(H * r)
            orig = orig.resize((W, H), Image.LANCZOS)

        arr_check = np.array(orig, dtype=np.uint8)
        orig = Image.fromarray(arr_check, 'RGBA')

        specs = []
        lid = 1

        bg = orig.copy().convert('RGB')
        bg_rgba = Image.new('RGBA', (W, H), (255, 255, 255, 255))
        bg_rgba.paste(bg, (0, 0))
        specs.append({
            'type': 'pixel', 'name': 'Background',
            'image': bg_rgba, 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

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
                    specs.append({
                        'type': 'pixel', 'name': 'Subject Masked',
                        'image': subj, 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
                    })
                    lid += 1
            except Exception as e:
                print('removebg:', e)

        specs.append({
            'type': 'adjustment', 'name': 'Curves 1',
            'adj_block': make_curv_block(),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        specs.append({
            'type': 'adjustment', 'name': 'Brightness/Contrast 1',
            'adj_block': make_brit_block(20, 10),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        specs.append({
            'type': 'adjustment', 'name': 'Hue/Saturation 1',
            'adj_block': make_hue2_block(0, 15, 5),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        specs.append({
            'type': 'adjustment', 'name': 'Color Balance 1',
            'adj_block': make_blnc_block(-10, 5, 15),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        specs.append({
            'type': 'adjustment', 'name': 'Levels 1',
            'adj_block': make_levl_block(),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

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
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
