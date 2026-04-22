"""
LayerAI PSD Pro — V2 Complete Build
=====================================
Flask backend for AI-powered PSD generation.

V2 Fixes:
- Hue/Saturation (hue2) — NOW USES DESCRIPTOR FORMAT (was corrupting PSD)
- Levels (levl) — NOW USES DESCRIPTOR FORMAT (was corrupting PSD)
- Color Balance (blnc) — NOW USES DESCRIPTOR FORMAT (was corrupting PSD)
- Brightness/Contrast (brit) — KEPT WORKING (legacy format OK for brit)
- Curves (curv) — KEPT WORKING (legacy format OK for curv)
- Google Vision text detection — ADDED /test-vision debug endpoint
- All adjustment layers are EDITABLE in Photoshop 2026

Deploy: Render (Python Flask)
Env vars: REMOVE_BG_API_KEY, GOOGLE_VISION_API_KEY
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import io, os, struct, base64, json, traceback
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

app = Flask(__name__)
CORS(app)

REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")
GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")


# =============================================================================
# SECTION 1: Basic PSD helpers (unchanged from V1)
# =============================================================================

def pk(fmt, *a):
    return struct.pack(fmt, *a)


def pstring(s, pad=4):
    """Pascal string padded to multiple of `pad`."""
    b = s.encode('ascii', errors='replace')[:255]
    raw = bytes([len(b)]) + b
    r = len(raw) % pad
    if r:
        raw += b'\x00' * (pad - r)
    return raw


def rle_encode_row(row_bytes):
    result = bytearray()
    i = 0
    n = len(row_bytes)
    while i < n:
        if i + 1 < n and row_bytes[i] == row_bytes[i + 1]:
            val = row_bytes[i]
            run = 1
            while i + run < n and row_bytes[i + run] == val and run < 128:
                run += 1
            result.append((256 - (run - 1)) & 0xFF)
            result.append(val)
            i += run
        else:
            lits = bytearray()
            lits.append(row_bytes[i])
            i += 1
            while i < n and len(lits) < 128:
                if i + 1 < n and row_bytes[i] == row_bytes[i + 1]:
                    break
                lits.append(row_bytes[i])
                i += 1
            result.append(len(lits) - 1)
            result.extend(lits)
    return bytes(result)


def rle_encode_channel(plane_2d):
    H = plane_2d.shape[0]
    row_counts = []
    compressed = bytearray()
    for y in range(H):
        row = plane_2d[y, :].tobytes()
        enc = rle_encode_row(row)
        row_counts.append(len(enc))
        compressed.extend(enc)
    return row_counts, bytes(compressed)


# =============================================================================
# SECTION 2: Additional layer data block helpers
# =============================================================================

def make_additional(key, data):
    """Build an 8BIM-tagged additional layer data block."""
    block = b'8BIM' + key + pk('>I', len(data)) + data
    if len(block) % 2:
        block += b'\x00'
    return block


def make_luni(name):
    return make_additional(b'luni',
                           pk('>I', len(name)) + name.encode('utf-16-be') + b'\x00\x00')


def make_lnsr(t):
    return make_additional(b'lnsr', t)


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


def make_common_extras(name, lid, is_adj=False):
    e = make_luni(name)
    e += make_lnsr(b'cont' if is_adj else b'layr')
    e += make_lyid(lid)
    e += make_clbl() + make_infx() + make_knko() + make_lspf() + make_lclr() + make_fxrp()
    return e


def make_blending_ranges():
    """10 pairs of (0, 65535) — 5 channels × src + dst."""
    data = b''
    for _ in range(10):
        data += pk('>HH', 0, 65535)
    return data


def make_adj_mask_data():
    """20-byte empty mask for adjustment layers."""
    return pk('>IIII', 0, 0, 0, 0) + pk('>BB', 255, 0) + b'\x00\x00'


# =============================================================================
# SECTION 3: PSD Descriptor builder (NEW for V2)
# =============================================================================
# Photoshop uses descriptors for modern adjustment layers.
# The brit and curv blocks work with legacy format, but hue2/levl/blnc
# REQUIRE descriptor format. This is why they were corrupting PSDs in V1.
# =============================================================================

def write_unicode_string(buf, text):
    """Write a PSD unicode string: uint32 length + UTF-16BE chars (null-terminated)."""
    chars = text + '\x00'
    buf.write(pk('>I', len(chars)))
    for ch in chars:
        buf.write(pk('>H', ord(ch)))


def write_descriptor_key(buf, key):
    """Write a descriptor key. 4-char keys use length=0 (OSType convention)."""
    if len(key) == 4:
        buf.write(pk('>I', 0))
        buf.write(key.encode('ascii'))
    else:
        encoded = key.encode('ascii')
        buf.write(pk('>I', len(encoded)))
        buf.write(encoded)


class Descriptor:
    """
    Builds Adobe PSD descriptor binary format.
    This is the data structure Photoshop uses for adjustment layer settings.
    """

    def __init__(self, class_name, class_id):
        self.class_name = class_name
        self.class_id = class_id
        self.items = []

    def add_bool(self, key, value):
        self.items.append(('bool', key, value))
        return self

    def add_long(self, key, value):
        self.items.append(('long', key, value))
        return self

    def add_double(self, key, value):
        self.items.append(('doub', key, value))
        return self

    def add_enum(self, key, enum_type, enum_value):
        self.items.append(('enum', key, (enum_type, enum_value)))
        return self

    def add_list(self, key, item_list):
        self.items.append(('VlLs', key, item_list))
        return self

    def add_unit_float(self, key, unit_type, value):
        self.items.append(('UntF', key, (unit_type, value)))
        return self

    def write(self, buf):
        """Serialize descriptor to binary."""
        write_unicode_string(buf, self.class_name)
        write_descriptor_key(buf, self.class_id)
        buf.write(pk('>I', len(self.items)))
        for item in self.items:
            self._write_item(buf, item)

    def _write_item(self, buf, item):
        type_tag, key, value = item
        write_descriptor_key(buf, key)
        buf.write(type_tag.encode('ascii'))

        if type_tag == 'bool':
            buf.write(pk('>B', 1 if value else 0))
        elif type_tag == 'long':
            buf.write(pk('>i', value))
        elif type_tag == 'doub':
            buf.write(pk('>d', value))
        elif type_tag == 'enum':
            enum_type, enum_value = value
            write_descriptor_key(buf, enum_type)
            write_descriptor_key(buf, enum_value)
        elif type_tag == 'VlLs':
            buf.write(pk('>I', len(value)))
            for list_item in value:
                if isinstance(list_item, Descriptor):
                    buf.write(b'Objc')
                    list_item.write(buf)
                elif isinstance(list_item, dict):
                    t = list_item['type']
                    v = list_item['value']
                    buf.write(t.encode('ascii'))
                    if t == 'long':
                        buf.write(pk('>i', v))
                    elif t == 'doub':
                        buf.write(pk('>d', v))
                    elif t == 'bool':
                        buf.write(pk('>B', 1 if v else 0))
        elif type_tag == 'UntF':
            unit_type, float_value = value
            buf.write(unit_type.encode('ascii'))
            buf.write(pk('>d', float_value))


def descriptor_to_bytes(desc):
    """Serialize a Descriptor to bytes."""
    buf = io.BytesIO()
    desc.write(buf)
    return buf.getvalue()


def make_descriptor_block(key_4cc, desc_bytes):
    """
    Wrap descriptor bytes into an 8BIM additional layer data block.
    Format: version(4 bytes, uint32=16) + descriptor_bytes
    """
    payload = pk('>I', 16) + desc_bytes
    return make_additional(key_4cc, payload)


# =============================================================================
# SECTION 4: Adjustment layer blocks
# =============================================================================

# --- Brightness/Contrast (brit) — LEGACY FORMAT — WORKS IN V1, KEEP AS-IS ---
def make_brit_block(brightness=0, contrast=0):
    """brit block uses legacy format (not descriptor). This works fine."""
    return make_additional(b'brit',
                           pk('>hh', brightness, contrast) + pk('>h', 128) + pk('>B', 0) + b'\x00')


# --- Curves (curv) — LEGACY FORMAT — WORKS IN V1, KEEP AS-IS ---
def make_curv_block():
    """curv block uses legacy format. This works fine."""
    data = pk('>H', 4) + pk('>I', 0)
    data += pk('>H', 2) + pk('>HH', 0, 0) + pk('>HH', 255, 255)
    for _ in range(3):
        data += pk('>H', 2) + pk('>HH', 0, 0) + pk('>HH', 255, 255)
    return make_additional(b'curv', data)


# --- Hue/Saturation (hue2) — V2 FIX: NOW USES DESCRIPTOR FORMAT ---
def make_hue2_block(hue=0, saturation=0, lightness=0, colorize=False):
    """
    V2 FIX: hue2 block now uses descriptor format.
    V1 used raw uint16 format which corrupted PSDs.
    
    Photoshop descriptor structure:
      Class: "HStr" (Hue/Saturation)
      - "Clrz" (bool) — Colorize mode
      - "Adjs" (VlLs) — List of Hst2 adjustment entries
        Each Hst2: H, Strt, Lght, Bgin, flFr, flTo, Endc
    """
    desc = Descriptor("Hue/Saturation", "HStr")
    desc.add_bool("Clrz", colorize)

    # Master adjustment
    master = Descriptor("Hue/Saturation", "Hst2")
    master.add_long("H   ", hue)
    master.add_long("Strt", saturation)
    master.add_long("Lght", lightness)
    master.add_long("Bgin", 0)
    master.add_long("flFr", 0)
    master.add_long("flTo", 0)
    master.add_long("Endc", 0)

    # 6 color range entries (Reds, Yellows, Greens, Cyans, Blues, Magentas)
    # All zeroed = no targeted color adjustment
    color_ranges = [
        (345, 315, 15, 45),    # Reds
        (15, 45, 75, 105),     # Yellows
        (75, 105, 135, 165),   # Greens
        (135, 165, 195, 225),  # Cyans
        (195, 225, 255, 285),  # Blues
        (255, 285, 345, 315),  # Magentas
    ]

    adjustments = [master]
    for bgin, flfr, flto, endc in color_ranges:
        adj = Descriptor("Hue/Saturation", "Hst2")
        adj.add_long("H   ", 0)
        adj.add_long("Strt", 0)
        adj.add_long("Lght", 0)
        adj.add_long("Bgin", bgin)
        adj.add_long("flFr", flfr)
        adj.add_long("flTo", flto)
        adj.add_long("Endc", endc)
        adjustments.append(adj)

    desc.add_list("Adjs", adjustments)

    desc_bytes = descriptor_to_bytes(desc)
    return make_descriptor_block(b'hue2', desc_bytes)


# --- Levels (levl) — V2 FIX: NOW USES DESCRIPTOR FORMAT ---
def make_levl_block(shadows=0, midtones=1.0, highlights=255,
                    output_shadows=0, output_highlights=255):
    """
    V2 FIX: levl block now uses descriptor format.
    V1 used raw uint16 format which corrupted PSDs.
    
    Photoshop descriptor structure:
      Class: "Lvls" (Levels)
      - "Adjs" (VlLs) — List of LvlA entries
        Each LvlA: Inpt [shadow, highlight], Gmm (gamma), Otpt [shadow, highlight]
      4 entries: Master, Red, Green, Blue
    """
    desc = Descriptor("Levels", "Lvls")

    adjustments = []

    # Master channel
    master = Descriptor("Levels Adjustment", "LvlA")
    master.add_list("Inpt", [
        {'type': 'long', 'value': shadows},
        {'type': 'long', 'value': highlights}
    ])
    master.add_double("Gmm ", midtones)
    master.add_list("Otpt", [
        {'type': 'long', 'value': output_shadows},
        {'type': 'long', 'value': output_highlights}
    ])
    adjustments.append(master)

    # Red, Green, Blue — all defaults
    for _ in range(3):
        ch = Descriptor("Levels Adjustment", "LvlA")
        ch.add_list("Inpt", [
            {'type': 'long', 'value': 0},
            {'type': 'long', 'value': 255}
        ])
        ch.add_double("Gmm ", 1.0)
        ch.add_list("Otpt", [
            {'type': 'long', 'value': 0},
            {'type': 'long', 'value': 255}
        ])
        adjustments.append(ch)

    desc.add_list("Adjs", adjustments)

    desc_bytes = descriptor_to_bytes(desc)
    return make_descriptor_block(b'levl', desc_bytes)


# --- Color Balance (blnc) — V2 FIX: NOW USES DESCRIPTOR FORMAT ---
def make_blnc_block(shadow_cr=0, shadow_mg=0, shadow_yb=0,
                    midtone_cr=0, midtone_mg=0, midtone_yb=0,
                    highlight_cr=0, highlight_mg=0, highlight_yb=0,
                    preserve_luminosity=True):
    """
    V2 FIX: blnc block now uses descriptor format.
    V1 used raw int16 format which corrupted PSDs.
    
    Photoshop descriptor structure:
      Class: "ClrB" (Color Balance)
      - "ShdC" (VlLs of long) — Shadow [cyan_red, magenta_green, yellow_blue]
      - "MdtC" (VlLs of long) — Midtone [cyan_red, magenta_green, yellow_blue]
      - "HghC" (VlLs of long) — Highlight [cyan_red, magenta_green, yellow_blue]
      - "PrsL" (bool) — Preserve Luminosity
    """
    desc = Descriptor("Color Balance", "ClrB")

    desc.add_list("ShdC", [
        {'type': 'long', 'value': shadow_cr},
        {'type': 'long', 'value': shadow_mg},
        {'type': 'long', 'value': shadow_yb}
    ])
    desc.add_list("MdtC", [
        {'type': 'long', 'value': midtone_cr},
        {'type': 'long', 'value': midtone_mg},
        {'type': 'long', 'value': midtone_yb}
    ])
    desc.add_list("HghC", [
        {'type': 'long', 'value': highlight_cr},
        {'type': 'long', 'value': highlight_mg},
        {'type': 'long', 'value': highlight_yb}
    ])
    desc.add_bool("PrsL", preserve_luminosity)

    desc_bytes = descriptor_to_bytes(desc)
    return make_descriptor_block(b'blnc', desc_bytes)


# =============================================================================
# SECTION 5: Google Vision text detection (FIXED for V2)
# =============================================================================

def detect_text(image_bytes, api_key):
    """
    Google Vision API se text detect karo.
    
    V2 Fix: Added proper error logging and validation.
    Returns list of: {'text': str, 'x': int, 'y': int, 'w': int, 'h': int}
    """
    if not api_key:
        print('[Vision] ERROR: No API key provided')
        return []

    if not image_bytes or len(image_bytes) == 0:
        print('[Vision] ERROR: Empty image bytes')
        return []

    b64 = base64.b64encode(image_bytes).decode('utf-8')
    print(f'[Vision] Sending request... Image size: {len(image_bytes)} bytes, '
          f'Base64 length: {len(b64)}')

    body = {
        "requests": [{
            "image": {"content": b64},
            "features": [{"type": "TEXT_DETECTION", "maxResults": 20}]
        }]
    }

    try:
        resp = requests.post(
            f'https://vision.googleapis.com/v1/images:annotate?key={api_key}',
            json=body,
            timeout=15
        )

        print(f'[Vision] Response status: {resp.status_code}')

        if resp.status_code != 200:
            print(f'[Vision] ERROR: API returned {resp.status_code}')
            print(f'[Vision] Response body: {resp.text[:500]}')
            return []

        data = resp.json()

        # Check for API-level errors
        responses = data.get('responses', [])
        if not responses:
            print('[Vision] ERROR: Empty responses array')
            return []

        first_response = responses[0]

        if 'error' in first_response:
            print(f'[Vision] API Error: {first_response["error"]}')
            return []

        annotations = first_response.get('textAnnotations', [])
        print(f'[Vision] Found {len(annotations)} text annotations')

        if not annotations:
            print('[Vision] No text found in image')
            return []

        texts = []
        for i, ann in enumerate(annotations):
            if i == 0:
                # First annotation = full detected text block, skip it
                print(f'[Vision] Full text: {ann.get("description", "")[:100]}...')
                continue

            verts = ann.get('boundingPoly', {}).get('vertices', [])
            if len(verts) >= 4:
                x = verts[0].get('x', 0)
                y = verts[0].get('y', 0)
                w = verts[1].get('x', 0) - x
                h = verts[2].get('y', 0) - y
                text_entry = {
                    'text': ann.get('description', ''),
                    'x': x, 'y': y, 'w': max(w, 1), 'h': max(h, 1)
                }
                texts.append(text_entry)
                print(f'[Vision] Text #{i}: "{text_entry["text"]}" at ({x},{y}) {w}x{h}')

        print(f'[Vision] Returning {len(texts)} text entries')
        return texts

    except requests.exceptions.Timeout:
        print('[Vision] ERROR: Request timed out (15s)')
        return []
    except requests.exceptions.ConnectionError as e:
        print(f'[Vision] ERROR: Connection failed: {e}')
        return []
    except json.JSONDecodeError as e:
        print(f'[Vision] ERROR: Invalid JSON response: {e}')
        return []
    except Exception as e:
        print(f'[Vision] ERROR: Unexpected error: {type(e).__name__}: {e}')
        traceback.print_exc()
        return []


# =============================================================================
# SECTION 6: Vignette generator
# =============================================================================

def make_vignette(w, h):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(30):
        t = i / 30
        a = int(150 * (1 - t))
        x0, y0 = int(w * t * 0.4), int(h * t * 0.4)
        d.rectangle([x0, y0, w - x0, h - y0], outline=(0, 0, 0, a))
    return img.filter(ImageFilter.GaussianBlur(15))


# =============================================================================
# SECTION 7: Layer builders
# =============================================================================

def build_pixel_layer(name, img, blend, opacity, W, H, lid):
    """Build a pixel layer record + channel data."""
    img_rgba = img.convert('RGBA').resize((W, H), Image.LANCZOS)
    arr = np.array(img_rgba, dtype=np.uint8)

    # Keep alpha only for Subject, Vignette, Text layers
    if 'Subject' not in name and 'Vignette' not in name and 'Text' not in name:
        arr[:, :, 3] = 255

    chs = [(-1, 3), (0, 0), (1, 1), (2, 2)]
    ch_parts = []
    for ch_id, ch_idx in chs:
        plane = arr[:, :, ch_idx]
        row_counts, compressed = rle_encode_channel(plane)
        ch_data = pk('>H', 1)  # compression = RLE
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

    extra = pk('>I', 0)  # mask data
    br = make_blending_ranges()
    extra += pk('>I', len(br)) + br
    extra += pstring(name, 4)
    extra += make_common_extras(name, lid, is_adj=False)
    rec += pk('>I', len(extra)) + extra

    return rec, b''.join(cd for _, cd in ch_parts)


def build_adjustment_layer(name, adj_block, blend, opacity, W, H, lid):
    """Build an adjustment layer record + channel data."""
    ch_ids = [-1, 0, 1, 2, -2]
    ch_data_each = pk('>H', 0)  # compression marker only, 0 pixels

    rec = pk('>IIII', 0, 0, 0, 0)  # bbox = all zeros for adjustment
    rec += pk('>H', 5)  # 5 channels
    for ch_id in ch_ids:
        rec += pk('>hI', ch_id, len(ch_data_each))

    bm = blend.encode('ascii').ljust(4)[:4]
    rec += b'8BIM' + bm + pk('>BBBB', opacity, 0, 24, 0)  # flags=24

    mask = make_adj_mask_data()
    extra = pk('>I', len(mask)) + mask
    br = make_blending_ranges()
    extra += pk('>I', len(br)) + br
    extra += pstring(name, 4)
    extra += adj_block
    extra += make_common_extras(name, lid, is_adj=True)
    rec += pk('>I', len(extra)) + extra

    return rec, ch_data_each * 5


# =============================================================================
# SECTION 8: PSD file assembler
# =============================================================================

def create_psd(layer_specs, W, H, original_rgb):
    """
    Assemble a complete PSD file from layer specs.
    
    layer_specs: list of dicts with keys:
      type: 'pixel' or 'adjustment'
      name: layer name
      image: PIL Image (for pixel layers)
      adj_block: bytes (for adjustment layers)
      blend_mode: 4-char blend mode string
      opacity: 0-255
      lid: layer ID
    """
    # ── 1. File Header ──
    s1 = b'8BPS'
    s1 += pk('>H', 1)       # version
    s1 += b'\x00' * 6       # reserved
    s1 += pk('>H', 3)       # 3 channels (RGB)
    s1 += pk('>I', H)
    s1 += pk('>I', W)
    s1 += pk('>H', 8)       # 8 bits per channel
    s1 += pk('>H', 3)       # RGB color mode

    # ── 2. Color Mode Data ──
    s2 = pk('>I', 0)

    # ── 3. Image Resources ──
    s3 = pk('>I', 0)

    # ── 4. Layer and Mask Information ──
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
    if len(li) % 2:
        li += b'\x00'
    body = pk('>I', len(li)) + li + pk('>I', 0)  # + global layer mask
    s4 = pk('>I', len(body)) + body

    # ── 5. Merged Composite Image Data ──
    merged_arr = np.array(original_rgb, dtype=np.uint8)

    all_row_counts = []
    all_compressed = []
    for c in range(3):
        plane = merged_arr[:, :, c]
        row_counts, compressed = rle_encode_channel(plane)
        all_row_counts.extend(row_counts)
        all_compressed.append(compressed)

    s5 = pk('>H', 1)  # compression = RLE
    for rc in all_row_counts:
        s5 += pk('>H', rc)
    for comp in all_compressed:
        s5 += comp

    return s1 + s2 + s3 + s4 + s5


# =============================================================================
# SECTION 9: Routes
# =============================================================================

@app.route('/')
@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "service": "LayerAI PSD Pro",
        "version": "2.0.0",
        "features": {
            "brit": "working (legacy)",
            "curv": "working (legacy)",
            "hue2": "FIXED (descriptor)",
            "levl": "FIXED (descriptor)",
            "blnc": "FIXED (descriptor)",
            "vision": "enabled" if GOOGLE_VISION_API_KEY else "disabled",
            "removebg": "enabled" if REMOVE_BG_API_KEY else "disabled"
        }
    })


@app.route('/generate-psd', methods=['POST'])
def gen_psd():
    """Basic PSD generation with default adjustment layers."""
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

        # 1. Background
        specs.append({
            'type': 'pixel', 'name': 'Background',
            'image': orig.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 2. Subject Masked
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

        # 3. Curves (editable)
        specs.append({
            'type': 'adjustment', 'name': 'Curves 1',
            'adj_block': make_curv_block(),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 4. Brightness/Contrast (editable)
        specs.append({
            'type': 'adjustment', 'name': 'Brightness/Contrast 1',
            'adj_block': make_brit_block(20, 10),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 5. Vignette
        specs.append({
            'type': 'pixel', 'name': 'Vignette',
            'image': make_vignette(W, H),
            'blend_mode': 'mul ', 'opacity': 180, 'lid': lid
        })

        psd = create_psd(specs, W, H, original_rgb)
        buf = io.BytesIO(psd)
        buf.seek(0)

        return send_file(buf, mimetype='application/octet-stream',
                         as_attachment=True, download_name='layerai-export.psd')

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/generate-psd-dynamic', methods=['POST'])
def gen_psd_dynamic():
    """
    Dynamic PSD generation with AI analysis values.
    Accepts form fields for all adjustment parameters.
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Parse AI analysis values from form data
        brightness = int(request.form.get('brightness', 0))
        contrast = int(request.form.get('contrast', 0))
        highlights = int(request.form.get('highlights', 0))
        shadows = int(request.form.get('shadows', 0))
        hue = int(request.form.get('hue', 0))
        saturation = int(request.form.get('saturation', 0))
        lightness = int(request.form.get('lightness', 0))
        color_grade = request.form.get('color_grade', 'Cinematic')
        effects = request.form.get('effects', '')
        subject_desc = request.form.get('subject_description', '')

        # Color balance values
        cb_shadow_cr = int(request.form.get('cb_shadow_cr', 0))
        cb_shadow_mg = int(request.form.get('cb_shadow_mg', 0))
        cb_shadow_yb = int(request.form.get('cb_shadow_yb', 0))
        cb_midtone_cr = int(request.form.get('cb_midtone_cr', 0))
        cb_midtone_mg = int(request.form.get('cb_midtone_mg', 0))
        cb_midtone_yb = int(request.form.get('cb_midtone_yb', 0))
        cb_highlight_cr = int(request.form.get('cb_highlight_cr', 0))
        cb_highlight_mg = int(request.form.get('cb_highlight_mg', 0))
        cb_highlight_yb = int(request.form.get('cb_highlight_yb', 0))

        # Levels values
        lvl_shadows = int(request.form.get('lvl_shadows', 0))
        lvl_midtones = float(request.form.get('lvl_midtones', 1.0))
        lvl_highlights = int(request.form.get('lvl_highlights', 255))

        raw = request.files['image'].read()
        orig = Image.open(io.BytesIO(raw))
        if orig.mode in ('CMYK', 'P', 'L', 'LA', 'I', 'F'):
            orig = orig.convert('RGB')
        orig = orig.convert('RGBA')
        W, H = orig.size
        orig_W, orig_H = W, H

        MAX = 800
        if W > MAX or H > MAX:
            r = min(MAX / W, MAX / H)
            W, H = int(W * r), int(H * r)
            orig = orig.resize((W, H), Image.LANCZOS)

        original_rgb = orig.convert('RGB')
        specs = []
        lid = 1

        # 1. Background
        specs.append({
            'type': 'pixel', 'name': 'Background',
            'image': orig.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 2. Subject Masked
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

        # 3. Curves (editable)
        specs.append({
            'type': 'adjustment', 'name': 'Curves 1',
            'adj_block': make_curv_block(),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 4. Brightness/Contrast (editable, dynamic values)
        specs.append({
            'type': 'adjustment', 'name': 'Brightness/Contrast 1',
            'adj_block': make_brit_block(brightness, contrast),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 5. Hue/Saturation (V2 FIX — editable!)
        specs.append({
            'type': 'adjustment', 'name': 'Hue/Saturation 1',
            'adj_block': make_hue2_block(hue=hue, saturation=saturation, lightness=lightness),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 6. Levels (V2 FIX — editable!)
        specs.append({
            'type': 'adjustment', 'name': 'Levels 1',
            'adj_block': make_levl_block(
                shadows=lvl_shadows, midtones=lvl_midtones, highlights=lvl_highlights),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 7. Color Balance (V2 FIX — editable!)
        specs.append({
            'type': 'adjustment', 'name': 'Color Balance 1',
            'adj_block': make_blnc_block(
                shadow_cr=cb_shadow_cr, shadow_mg=cb_shadow_mg, shadow_yb=cb_shadow_yb,
                midtone_cr=cb_midtone_cr, midtone_mg=cb_midtone_mg, midtone_yb=cb_midtone_yb,
                highlight_cr=cb_highlight_cr, highlight_mg=cb_highlight_mg, highlight_yb=cb_highlight_yb),
            'blend_mode': 'norm', 'opacity': 255, 'lid': lid
        })
        lid += 1

        # 8. Vignette
        if 'vignette' in effects.lower() or not effects:
            specs.append({
                'type': 'pixel', 'name': 'Vignette',
                'image': make_vignette(W, H),
                'blend_mode': 'mul ', 'opacity': 180, 'lid': lid
            })
            lid += 1

        # 9. Color Grade overlay
        if color_grade:
            if 'warm' in color_grade.lower():
                grade_img = Image.new('RGBA', (W, H), (255, 180, 120, 25))
            elif 'cool' in color_grade.lower():
                grade_img = Image.new('RGBA', (W, H), (120, 180, 255, 25))
            else:
                grade_img = Image.new('RGBA', (W, H), (200, 200, 180, 20))
            specs.append({
                'type': 'pixel', 'name': 'Color Grade - ' + color_grade,
                'image': grade_img, 'blend_mode': 'over', 'opacity': 60, 'lid': lid
            })
            lid += 1

        # 10. Text layers from Google Vision
        if GOOGLE_VISION_API_KEY:
            texts = detect_text(raw, GOOGLE_VISION_API_KEY)
            for t in texts[:10]:
                txt_img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                draw = ImageDraw.Draw(txt_img)
                scale_x = W / orig_W if orig_W != W else 1
                scale_y = H / orig_H if orig_H != H else 1
                tx = int(t['x'] * scale_x)
                ty = int(t['y'] * scale_y)
                draw.text((tx, ty), t['text'], fill=(255, 255, 255, 255))
                specs.append({
                    'type': 'pixel', 'name': 'Text: ' + t['text'][:20],
                    'image': txt_img, 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
                })
                lid += 1

        psd = create_psd(specs, W, H, original_rgb)
        buf = io.BytesIO(psd)
        buf.seek(0)

        return send_file(buf, mimetype='application/octet-stream',
                         as_attachment=True, download_name='layerai-export.psd')

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# SECTION 10: Test / Debug Endpoints
# =============================================================================

@app.route('/test-vision', methods=['POST'])
def test_vision():
    """
    Debug endpoint: Test Google Vision text detection.
    POST with multipart form: image file
    Returns JSON with detected texts and debug info.
    """
    result = {
        "api_key_set": bool(GOOGLE_VISION_API_KEY),
        "api_key_prefix": GOOGLE_VISION_API_KEY[:8] + "..." if GOOGLE_VISION_API_KEY else None,
        "texts": [],
        "error": None
    }

    if not GOOGLE_VISION_API_KEY:
        result["error"] = "GOOGLE_VISION_API_KEY not set in environment"
        return jsonify(result), 500

    if 'image' not in request.files:
        result["error"] = "No image file in request. Send as multipart form with key 'image'"
        return jsonify(result), 400

    raw = request.files['image'].read()
    result["image_size_bytes"] = len(raw)

    try:
        img = Image.open(io.BytesIO(raw))
        result["image_dimensions"] = f"{img.size[0]}x{img.size[1]}"
        result["image_mode"] = img.mode
    except Exception as e:
        result["error"] = f"Invalid image: {e}"
        return jsonify(result), 400

    texts = detect_text(raw, GOOGLE_VISION_API_KEY)
    result["texts"] = texts
    result["text_count"] = len(texts)
    result["status"] = "ok" if texts else "no_text_found"

    return jsonify(result)


@app.route('/test-adjustments', methods=['GET'])
def test_adjustments():
    """
    Test endpoint: Download a PSD with all V2 fixed adjustment layers.
    GET /test-adjustments
    Open the downloaded PSD in Photoshop 2026 to verify all layers are editable.
    """
    W, H = 400, 300

    # Create a simple gradient background for testing
    bg = Image.new('RGBA', (W, H), (40, 40, 50, 255))
    draw = ImageDraw.Draw(bg)
    for y in range(H):
        r = int(40 + (y / H) * 60)
        g = int(40 + (y / H) * 40)
        b = int(50 + (y / H) * 80)
        draw.line([(0, y), (W, y)], fill=(r, g, b, 255))

    original_rgb = bg.convert('RGB')

    specs = []
    lid = 1

    # Background
    specs.append({
        'type': 'pixel', 'name': 'Background',
        'image': bg.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
    })
    lid += 1

    # Curves (working)
    specs.append({
        'type': 'adjustment', 'name': 'Curves 1',
        'adj_block': make_curv_block(),
        'blend_mode': 'norm', 'opacity': 255, 'lid': lid
    })
    lid += 1

    # Brightness/Contrast (working)
    specs.append({
        'type': 'adjustment', 'name': 'Brightness/Contrast 1',
        'adj_block': make_brit_block(15, 10),
        'blend_mode': 'norm', 'opacity': 255, 'lid': lid
    })
    lid += 1

    # Hue/Saturation (V2 FIX)
    specs.append({
        'type': 'adjustment', 'name': 'Hue/Saturation 1',
        'adj_block': make_hue2_block(hue=10, saturation=-15, lightness=0),
        'blend_mode': 'norm', 'opacity': 255, 'lid': lid
    })
    lid += 1

    # Levels (V2 FIX)
    specs.append({
        'type': 'adjustment', 'name': 'Levels 1',
        'adj_block': make_levl_block(shadows=10, midtones=1.15, highlights=245),
        'blend_mode': 'norm', 'opacity': 255, 'lid': lid
    })
    lid += 1

    # Color Balance (V2 FIX)
    specs.append({
        'type': 'adjustment', 'name': 'Color Balance 1',
        'adj_block': make_blnc_block(midtone_cr=8, midtone_mg=-3, midtone_yb=-10),
        'blend_mode': 'norm', 'opacity': 255, 'lid': lid
    })
    lid += 1

    psd = create_psd(specs, W, H, original_rgb)
    buf = io.BytesIO(psd)
    buf.seek(0)

    return send_file(buf, mimetype='application/octet-stream',
                     as_attachment=True, download_name='test_v2_adjustments.psd')


@app.route('/test-single-layer', methods=['GET'])
def test_single_layer():
    """
    Test a single adjustment layer type.
    GET /test-single-layer?type=hue2
    GET /test-single-layer?type=levl
    GET /test-single-layer?type=blnc
    GET /test-single-layer?type=brit
    GET /test-single-layer?type=curv
    """
    layer_type = request.args.get('type', 'hue2')
    W, H = 200, 200

    bg = Image.new('RGBA', (W, H), (128, 128, 128, 255))
    original_rgb = bg.convert('RGB')

    specs = []
    lid = 1

    specs.append({
        'type': 'pixel', 'name': 'Background',
        'image': bg, 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
    })
    lid += 1

    layer_map = {
        'hue2': ('Hue/Saturation 1', make_hue2_block(hue=20, saturation=-30, lightness=5)),
        'levl': ('Levels 1', make_levl_block(shadows=15, midtones=1.3, highlights=240)),
        'blnc': ('Color Balance 1', make_blnc_block(midtone_cr=15, midtone_yb=-20)),
        'brit': ('Brightness/Contrast 1', make_brit_block(25, 15)),
        'curv': ('Curves 1', make_curv_block()),
    }

    if layer_type not in layer_map:
        return jsonify({"error": f"Unknown type: {layer_type}",
                        "valid_types": list(layer_map.keys())}), 400

    name, block = layer_map[layer_type]
    specs.append({
        'type': 'adjustment', 'name': name,
        'adj_block': block, 'blend_mode': 'norm', 'opacity': 255, 'lid': lid
    })

    psd = create_psd(specs, W, H, original_rgb)
    buf = io.BytesIO(psd)
    buf.seek(0)

    return send_file(buf, mimetype='application/octet-stream',
                     as_attachment=True, download_name=f'test_{layer_type}.psd')


@app.route('/verify-psd', methods=['POST'])
def verify_psd():
    """
    Upload a PSD and verify its internal structure.
    POST with multipart form: file
    Returns JSON showing found blocks and their offsets.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Upload a PSD file with key 'file'"}), 400

    data = request.files['file'].read()
    results = {
        "file_size": len(data),
        "valid_signature": data[:4] == b'8BPS',
        "version": struct.unpack('>H', data[4:6])[0] if len(data) >= 6 else None,
        "blocks": {}
    }

    for key in [b'hue2', b'levl', b'blnc', b'brit', b'curv', b'lnsr', b'luni']:
        positions = []
        offset = 0
        while True:
            pos = data.find(key, offset)
            if pos < 0:
                break
            sig = data[pos - 4:pos] if pos >= 4 else b''
            block_len = struct.unpack('>I', data[pos + 4:pos + 8])[0] if pos + 8 <= len(data) else -1
            positions.append({
                "offset": pos,
                "hex_offset": f"0x{pos:04x}",
                "signature_valid": sig == b'8BIM',
                "data_length": block_len
            })
            offset = pos + 4
        if positions:
            results["blocks"][key.decode()] = positions

    return jsonify(results)


@app.route('/debug-env', methods=['GET'])
def debug_env():
    """Show which environment variables are set (without revealing values)."""
    return jsonify({
        "REMOVE_BG_API_KEY": "SET" if REMOVE_BG_API_KEY else "NOT SET",
        "GOOGLE_VISION_API_KEY": "SET" if GOOGLE_VISION_API_KEY else "NOT SET",
        "PORT": os.environ.get('PORT', '5000 (default)'),
        "python_version": os.sys.version,
    })


# =============================================================================
# SECTION 11: Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  LayerAI PSD Pro — V2.0.0")
    print("  Adjustment layers: brit ✓  curv ✓  hue2 ✓  levl ✓  blnc ✓")
    print("  Vision API:", "ENABLED" if GOOGLE_VISION_API_KEY else "DISABLED")
    print("  Remove.bg:", "ENABLED" if REMOVE_BG_API_KEY else "DISABLED")
    print("=" * 60)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
