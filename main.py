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

def rle_encode_row(row_bytes):
    """PackBits RLE encode a single row of bytes"""
    result = bytearray()
    i = 0
    n = len(row_bytes)
    while i < n:
        # Check for run
        if i + 1 < n and row_bytes[i] == row_bytes[i+1]:
            val = row_bytes[i]
            run = 1
            while i + run < n and row_bytes[i+run] == val and run < 128:
                run += 1
            result.append((256 - (run - 1)) & 0xFF)
            result.append(val)
            i += run
        else:
            # Literal run
            lits = bytearray()
            lits.append(row_bytes[i])
            i += 1
            while i < n and len(lits) < 128:
                if i + 1 < n and row_bytes[i] == row_bytes[i+1]:
                    break
                lits.append(row_bytes[i])
                i += 1
            result.append(len(lits) - 1)
            result.extend(lits)
    return bytes(result)

def rle_encode_channel(plane_2d):
    """RLE encode a 2D numpy array, return (row_byte_counts, compressed_data)"""
    H = plane_2d.shape[0]
    row_counts = []
    compressed = bytearray()
    for y in range(H):
        row = plane_2d[y, :].tobytes()
        enc = rle_encode_row(row)
        row_counts.append(len(enc))
        compressed.extend(enc)
    return row_counts, bytes(compressed)

def make_additional(key, data):
    block = b'8BIM' + key + pk('>I', len(data)) + data
    if len(block) % 2: block += b'\x00'
    return block

def make_luni(name):
    return make_additional(b'luni', pk('>I', len(name)) + name.encode('utf-16-be') + b'\x00\x00')
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
    e += make_clbl() + make_infx() + make_knko() + make_lspf() + make_lclr() + make_fxrp()
    return e

def make_brit_block(brightness=0, contrast=0):
    return make_additional(b'brit', pk('>hh', brightness, contrast) + pk('>h', 128) + pk('>B', 0) + b'\x00')

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
    data = pk('>H', 2) + pk('>HHHHH', 0, 128, 255, 0, 255)
    for _ in range(3):
        data += pk('>HHHHH', 0, 128, 255, 0, 255)
    return make_additional(b'levl', data)

def make_blnc_block(cr=0, mg=0, yb=0):
    data = pk('>hhh', 0, 0, 0) + pk('>hhh', cr, mg, yb) + pk('>hhh', 0, 0, 0) + pk('>B', 1) + b'\x00'
    return make_additional(b'blnc', data)

def make_blending_ranges():
    data = b''
    for _ in range(10):
        data += pk('>HH', 0, 65535)
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

def build_pixel_layer(name, img, blend, opacity, W, H, lid):
    # Convert to RGBA, force alpha=255 for non-transparent layers
    img_rgba = img.convert('RGBA').resize((W, H), Image.LANCZOS)
    arr = np.array(img_rgba, dtype=np.uint8)
    
    # Force alpha=255 for Background
    if 'Subject' not in name and 'Vignette' not in name:
        arr[:, :, 3] = 255

    # 4 channels with RLE compression (like real Photoshop)
    chs = [(-1, 3), (0, 0), (1, 1), (2, 2)]
    ch_parts = []
    for ch_id, ch_idx in chs:
        plane = arr[:, :, ch_idx]
        row_counts, compressed = rle_encode_channel(plane)
        # Channel data: compression=1 + row byte counts + compressed rows
        ch_data = pk('>H', 1)
        for rc in row_counts:
            ch_data += pk('>H', rc)
        ch_data += compressed
        ch_parts.append((ch_id, ch_data))

    rec = pk('>IIII', 0, 0, H, W)
    rec += pk('>H', 4)
    for ch_id, ch_data in ch_parts:
        rec += pk('>hI', ch_id, len(ch_data))

    bm = blend.encode('ascii').ljust(4)[:4]
    rec += b'8BIM' + bm + pk('>BBBB', opacity, 0, 8, 0)

    extra = pk('>I', 0)
    br = make_blending_ranges()
    extra += pk('>I', len(br)) + br
    extra += pstring(name, 4)
    extra += make_common_extras(name, lid, is_adj=False)
    rec += pk('>I', len(extra)) + extra

    return rec, b''.join(cd for _, cd in ch_parts)

def build_adjustment_layer(name, adj_block, blend, opacity, W, H, lid):
    ch_ids = [-1, 0, 1, 2, -2]
    ch_data_each = pk('>H', 0)  # compression=0, no pixels (empty bbox)

    rec = pk('>IIII', 0, 0, 0, 0)
    rec += pk('>H', 5)
    for ch_id in ch_ids:
        rec += pk('>hI', ch_id, len(ch_data_each))

    bm = blend.encode('ascii').ljust(4)[:4]
    rec += b'8BIM' + bm + pk('>BBBB', opacity, 0, 24, 0)

    mask = make_adj_mask_data()
    extra = pk('>I', len(mask)) + mask
    br = make_blending_ranges()
    extra += pk('>I', len(br)) + br
    extra += pstring(name, 4)
    extra += adj_block
    extra += make_common_extras(name, lid, is_adj=True)
    rec += pk('>I', len(extra)) + extra

    return rec, ch_data_each * 5

def create_psd(layer_specs, W, H, original_rgb):
    # Header: 3 channels (like real Photoshop)
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

    li = pk('>h', len(layer_specs)) + all_records + all_chdata
    if len(li) % 2: li += b'\x00'
    body = pk('>I', len(li)) + li + pk('>I', 0)
    s4 = pk('>I', len(body)) + body

    # Merged composite with RLE (like real Photoshop)
    merged_arr = np.array(original_rgb, dtype=np.uint8)
    
    # RLE compress each channel
    all_row_counts = []
    all_compressed = []
    for c in range(3):
        plane = merged_arr[:, :, c]
        row_counts, compressed = rle_encode_channel(plane)
        all_row_counts.extend(row_counts)
        all_compressed.append(compressed)

    s5 = pk('>H', 1)  # RLE compression
    for rc in all_row_counts:
        s5 += pk('>H', rc)
    for comp in all_compressed:
        s5 += comp

    return s1 + s2 + s3 + s4 + s5

@app.route('/')
@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "LayerAI PSD Pro", "version": "13.0.0"})

@app.route('/generate-psd', methods=['POST'])
def gen_psd():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        raw = request.files['image'].read()
        orig = Image.open(io.BytesIO(raw))
        if orig.mode in ('CMYK', 'P', 'L', 'LA', 'I', 'F'):
            orig = orig.convert('RGB')
        orig = orig.convert('RGBA')
        W, H = orig.size

        MAX = 800
        if W > MAX or H > MAX:
            r = min(MAX / W, MAX / H)
            W, H = int(W * r), int(H * r)
            orig = orig.resize((W, H), Image.LANCZOS)

        original_rgb = orig.convert('RGB')

        specs = []
        lid = 1

        # Background
        specs.append({
            'type': 'pixel', 'name': 'Background',
            'image': orig.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # Subject
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

        specs.append({'type': 'adjustment', 'name': 'Curves 1',
            'adj_block': make_curv_block(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1
        specs.append({'type': 'adjustment', 'name': 'Brightness/Contrast 1',
            'adj_block': make_brit_block(20, 10), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1
        specs.append({'type': 'adjustment', 'name': 'Hue/Saturation 1',
            'adj_block': make_hue2_block(0, 15, 5), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1
        specs.append({'type': 'adjustment', 'name': 'Color Balance 1',
            'adj_block': make_blnc_block(-10, 5, 15), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1
        specs.append({'type': 'adjustment', 'name': 'Levels 1',
            'adj_block': make_levl_block(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1
        specs.append({
            'type': 'pixel', 'name': 'Vignette',
            'image': make_vignette(W, H), 'blend_mode': 'mul ', 'opacity': 180, 'lid': lid
        })

        psd = create_psd(specs, W, H, original_rgb)
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
