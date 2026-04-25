"""
LayerAI PSD Pro — V2.1 FINAL
==============================
ALL adjustment layers use LEGACY binary format (not descriptors).
Photoshop reads hue2/levl/blnc in legacy format only.

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
RAILWAY_API_URL = os.environ.get("RAILWAY_API_URL", "")

def pk(fmt, *a):
    return struct.pack(fmt, *a)

def pstring(s, pad=4):
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

# === Additional layer data blocks ===

def make_additional(key, data):
    block = b'8BIM' + key + pk('>I', len(data)) + data
    if len(block) % 2:
        block += b'\x00'
    return block

def make_luni(name):
    return make_additional(b'luni', pk('>I', len(name)) + name.encode('utf-16-be') + b'\x00\x00')

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
    data = b''
    for _ in range(10):
        data += pk('>HH', 0, 65535)
    return data

def make_adj_mask_data():
    return pk('>IIII', 0, 0, 0, 0) + pk('>BB', 255, 0) + b'\x00\x00'


# =============================================================================
# ADJUSTMENT LAYER BLOCKS — ALL LEGACY FORMAT
# =============================================================================

def make_brit_block(brightness=0, contrast=0):
    """Brightness/Contrast. Default (0,0) = no effect."""
    return make_additional(b'brit',
                           pk('>hh', brightness, contrast) +
                           pk('>h', 128) + pk('>B', 0) + b'\x00')

def make_curv_block():
    """Curves. Default straight line = no effect."""
    data = pk('>H', 4) + pk('>I', 0)
    for _ in range(4):
        data += pk('>H', 2) + pk('>HH', 0, 0) + pk('>HH', 255, 255)
    return make_additional(b'curv', data)

def make_hue2_block(hue=0, saturation=0, lightness=0, colorize=False):
    """
    Hue/Saturation — LEGACY FORMAT.
    
    Format: version(2) + colorize(1) + pad(1) + 7 entries × 14 bytes
    Each entry: 4 range boundaries (int16) + 3 adjustments (int16)
    Entry 0 = Master (range values all 0, adjustments = user values)
    Entry 1-6 = Color ranges (Reds, Yellows, Greens, Cyans, Blues, Magentas)
    Total: 4 + (7 × 14) = 102 bytes
    
    hue: -180 to 180, saturation: -100 to 100, lightness: -100 to 100
    Default (0,0,0) = no effect on image.
    """
    data = pk('>H', 2)                            # version = 2
    data += pk('>B', 1 if colorize else 0)         # colorize
    data += b'\x00'                                 # padding

    # Master: 4 range values (unused, all 0) + 3 adjustments
    data += pk('>hhhh', 0, 0, 0, 0)                # range (unused for master)
    data += pk('>hhh', hue, saturation, lightness)  # master adjustments

    # 6 color ranges: Reds, Yellows, Greens, Cyans, Blues, Magentas
    # Each: 4 range boundaries (int16) + 3 adjustments (int16) = 14 bytes
    default_ranges = [
        (315, 345, 15,  45),     # Reds
        (15,  45,  75,  105),    # Yellows
        (75,  105, 135, 165),    # Greens
        (135, 165, 195, 225),    # Cyans
        (195, 225, 255, 285),    # Blues
        (255, 285, 315, 345),    # Magentas
    ]
    for r1, r2, r3, r4 in default_ranges:
        data += pk('>hhhh', r1, r2, r3, r4)
        data += pk('>hhh', 0, 0, 0)

    return make_additional(b'hue2', data)

def make_levl_block(shadows=0, midtones=100, highlights=255,
                    output_shadows=0, output_highlights=255):
    """
    Levels — LEGACY FORMAT.
    
    Format: version(2) + 29 records × (input_floor, input_ceil, out_floor, out_ceil, gamma)
    Each value uint16. Gamma: 100=1.0, 50=0.5, 200=2.0
    Total: 292 bytes
    
    Default (0, 100, 255, 0, 255) = no effect on image.
    """
    data = pk('>H', 2)  # version

    # Master channel
    data += pk('>HHHHH', shadows, highlights, output_shadows, output_highlights, midtones)

    # R, G, B + 25 unused — all defaults
    for _ in range(28):
        data += pk('>HHHHH', 0, 255, 0, 255, 100)

    return make_additional(b'levl', data)

def make_blnc_block(shadow_cr=0, shadow_mg=0, shadow_yb=0,
                    midtone_cr=0, midtone_mg=0, midtone_yb=0,
                    highlight_cr=0, highlight_mg=0, highlight_yb=0,
                    preserve_luminosity=True):
    """
    Color Balance — LEGACY FORMAT.
    
    Format: shadows(6) + midtones(6) + highlights(6) + preserve_lum(1) + pad(1)
    Total: 20 bytes
    
    Values: -100 to 100. Default (all 0) = no effect on image.
    """
    data = pk('>hhh', shadow_cr, shadow_mg, shadow_yb)
    data += pk('>hhh', midtone_cr, midtone_mg, midtone_yb)
    data += pk('>hhh', highlight_cr, highlight_mg, highlight_yb)
    data += pk('>B', 1 if preserve_luminosity else 0)
    data += b'\x00'
    return make_additional(b'blnc', data)


# =============================================================================
# EDITABLE TEXT LAYER — TySh (Type Tool) block
# Template-based approach using exact binary from real Photoshop 2026 PSD
# =============================================================================

# Binary sections extracted from a real Photoshop text layer
# Section B: descriptor header (version 50 + class info + Txt key)
_TYSH_SEC_B = bytes.fromhex("0032000000100000000100000000000054784c72000000080000000054787420544558540000000c")
# Section D: textGridding + Ornt + AntA + bounds + boundingBox + TextIndex + EngineData key
_TYSH_SEC_D = bytes.fromhex("0000000c746578744772696464696e67656e756d0000000c746578744772696464696e67000000004e6f6e65000000004f726e74656e756d000000004f726e740000000048727a6e00000000416e7441656e756d00000000416e6e740000000e616e7469416c696173536861727000000006626f756e64734f626a6300000001000000000006626f756e647300000004000000004c656674556e744623506e74000000000000000000000000546f7020556e744623506e74c0467ff6000000000000000052676874556e744623506e7440697999a00000000000000042746f6d556e744623506e74402e0014000000000000000b626f756e64696e67426f784f626a630000000100000000000b626f756e64696e67426f7800000004000000004c656674556e744623506e743fc400000000000000000000546f7020556e744623506e74c0446c00000000000000000052676874556e744623506e74406977cce00000000000000042746f6d556e744623506e743feb0000000000000000000954657874496e6465786c6f6e67000000010000000a456e67696e65446174617464746100002218")
# Section F: warp descriptor + final bounds placeholder  
_TYSH_SEC_F = bytes.fromhex("00010000001000000001000000000004776172700000000500000009776172705374796c65656e756d00000009776172705374796c6500000008776172704e6f6e65000000097761727056616c7565646f756200000000000000000000000f776172705065727370656374697665646f75620000000000000000000000147761727050657273706563746976654f74686572646f756200000000000000000000000a77617270526f74617465656e756d000000004f726e740000000048727a6e00000000000000000000000000000000000000")


def _utf16be_escape(text_bytes):
    """Convert bytes to escaped ASCII string for EngineData parens."""
    result = ''
    for byte in text_bytes:
        if 32 <= byte < 127 and byte not in (ord('('), ord(')'), ord('\\')):
            result += chr(byte)
        else:
            result += f'\\x{byte:02x}'
    return result


def _build_engine_data(text, font_name='ArialMT', font_size=24.0,
                        r=1.0, g=1.0, b=1.0):
    """
    Build EngineData in PLAIN ASCII format (matching real Photoshop output).
    NOT UTF-16BE! That was the critical bug in previous versions.
    """
    # Text as UTF-16BE with BOM inside ASCII parens
    text_u16 = b'\xfe\xff' + text.encode('utf-16-be') + b'\x00\x0d'
    text_esc = _utf16be_escape(text_u16)

    # Font name as UTF-16BE with BOM
    font_u16 = b'\xfe\xff' + font_name.encode('utf-16-be')
    font_esc = _utf16be_escape(font_u16)

    # Fallback font
    fb_u16 = b'\xfe\xff' + 'MyriadPro-Regular'.encode('utf-16-be')
    fb_esc = _utf16be_escape(fb_u16)

    # "Normal RGB" label
    nrgb_u16 = b'\xfe\xff' + 'Normal RGB'.encode('utf-16-be')
    nrgb_esc = _utf16be_escape(nrgb_u16)

    # Kinsoku name
    kinsoku_u16 = b'\xfe\xff' + 'PhotoshopKinsokuHard'.encode('utf-16-be')
    kinsoku_esc = _utf16be_escape(kinsoku_u16)

    # default mojikumi
    moji_u16 = b'\xfe\xff' + 'default'.encode('utf-16-be')
    moji_esc = _utf16be_escape(moji_u16)

    tlen = len(text) + 1

    ed = (
        f'\n\n<<\n'
        f'\t/EngineDict\n'
        f'\t<<\n'
        f'\t\t/Editor\n'
        f'\t\t<<\n'
        f'\t\t\t/Text ({text_esc})\n'
        f'\t\t>>\n'
        f'\t\t/ParagraphRun\n'
        f'\t\t<<\n'
        f'\t\t\t/DefaultRunData\n'
        f'\t\t\t<<\n'
        f'\t\t\t\t/ParagraphSheet\n'
        f'\t\t\t\t<<\n'
        f'\t\t\t\t\t/DefaultStyleSheet 0\n'
        f'\t\t\t\t\t/Properties\n'
        f'\t\t\t\t\t<<\n'
        f'\t\t\t\t\t>>\n'
        f'\t\t\t\t>>\n'
        f'\t\t\t\t/Adjustments\n'
        f'\t\t\t\t<<\n'
        f'\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n'
        f'\t\t\t\t\t/XY [ 0.0 0.0 ]\n'
        f'\t\t\t\t>>\n'
        f'\t\t\t>>\n'
        f'\t\t\t/RunArray [\n'
        f'\t\t\t<<\n'
        f'\t\t\t\t/ParagraphSheet\n'
        f'\t\t\t\t<<\n'
        f'\t\t\t\t\t/DefaultStyleSheet 0\n'
        f'\t\t\t\t\t/Properties\n'
        f'\t\t\t\t\t<<\n'
        f'\t\t\t\t\t\t/Justification 0\n'
        f'\t\t\t\t\t\t/FirstLineIndent 0.0\n'
        f'\t\t\t\t\t\t/StartIndent 0.0\n'
        f'\t\t\t\t\t\t/EndIndent 0.0\n'
        f'\t\t\t\t\t\t/SpaceBefore 0.0\n'
        f'\t\t\t\t\t\t/SpaceAfter 0.0\n'
        f'\t\t\t\t\t\t/AutoHyphenate false\n'
        f'\t\t\t\t\t\t/HyphenatedWordSize 6\n'
        f'\t\t\t\t\t\t/PreHyphen 2\n'
        f'\t\t\t\t\t\t/PostHyphen 2\n'
        f'\t\t\t\t\t\t/ConsecutiveHyphens 8\n'
        f'\t\t\t\t\t\t/Zone 36.0\n'
        f'\t\t\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]\n'
        f'\t\t\t\t\t\t/LetterSpacing [ 0.0 0.0 0.0 ]\n'
        f'\t\t\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]\n'
        f'\t\t\t\t\t\t/AutoLeading 1.2\n'
        f'\t\t\t\t\t\t/LeadingType 0\n'
        f'\t\t\t\t\t\t/Hanging false\n'
        f'\t\t\t\t\t\t/Burasagari false\n'
        f'\t\t\t\t\t\t/KinsokuOrder 0\n'
        f'\t\t\t\t\t\t/EveryLineComposer false\n'
        f'\t\t\t\t\t>>\n'
        f'\t\t\t\t>>\n'
        f'\t\t\t\t/Adjustments\n'
        f'\t\t\t\t<<\n'
        f'\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]\n'
        f'\t\t\t\t\t/XY [ 0.0 0.0 ]\n'
        f'\t\t\t\t>>\n'
        f'\t\t\t>>\n'
        f'\t\t\t]\n'
        f'\t\t\t/RunLengthArray [ {tlen} ]\n'
        f'\t\t\t/IsJoinable 1\n'
        f'\t\t>>\n'
        f'\t\t/StyleRun\n'
        f'\t\t<<\n'
        f'\t\t\t/DefaultRunData\n'
        f'\t\t\t<<\n'
        f'\t\t\t\t/StyleSheet\n'
        f'\t\t\t\t<<\n'
        f'\t\t\t\t\t/StyleSheetData\n'
        f'\t\t\t\t\t<<\n'
        f'\t\t\t\t\t>>\n'
        f'\t\t\t\t>>\n'
        f'\t\t\t>>\n'
        f'\t\t\t/RunArray [\n'
        f'\t\t\t<<\n'
        f'\t\t\t\t/StyleSheet\n'
        f'\t\t\t\t<<\n'
        f'\t\t\t\t\t/StyleSheetData\n'
        f'\t\t\t\t\t<<\n'
        f'\t\t\t\t\t\t/Font 0\n'
        f'\t\t\t\t\t\t/FontSize {font_size:.1f}\n'
        f'\t\t\t\t\t\t/FauxBold false\n'
        f'\t\t\t\t\t\t/FauxItalic false\n'
        f'\t\t\t\t\t\t/AutoLeading true\n'
        f'\t\t\t\t\t\t/Leading .01\n'
        f'\t\t\t\t\t\t/HorizontalScale 1.0\n'
        f'\t\t\t\t\t\t/VerticalScale 1.0\n'
        f'\t\t\t\t\t\t/Tracking 0\n'
        f'\t\t\t\t\t\t/AutoKerning true\n'
        f'\t\t\t\t\t\t/Kerning 0\n'
        f'\t\t\t\t\t\t/BaselineShift 0.0\n'
        f'\t\t\t\t\t\t/FontCaps 0\n'
        f'\t\t\t\t\t\t/FontBaseline 0\n'
        f'\t\t\t\t\t\t/Underline false\n'
        f'\t\t\t\t\t\t/Strikethrough false\n'
        f'\t\t\t\t\t\t/Ligatures true\n'
        f'\t\t\t\t\t\t/DLigatures false\n'
        f'\t\t\t\t\t\t/BaselineDirection 1\n'
        f'\t\t\t\t\t\t/Tsume 0.0\n'
        f'\t\t\t\t\t\t/StyleRunAlignment 2\n'
        f'\t\t\t\t\t\t/Language 0\n'
        f'\t\t\t\t\t\t/NoBreak false\n'
        f'\t\t\t\t\t\t/FillColor\n'
        f'\t\t\t\t\t\t<<\n'
        f'\t\t\t\t\t\t\t/Type 1\n'
        f'\t\t\t\t\t\t\t/Values [ 1.0 {r:.4f} {g:.4f} {b:.4f} ]\n'
        f'\t\t\t\t\t\t>>\n'
        f'\t\t\t\t\t\t/StrokeColor\n'
        f'\t\t\t\t\t\t<<\n'
        f'\t\t\t\t\t\t\t/Type 1\n'
        f'\t\t\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]\n'
        f'\t\t\t\t\t\t>>\n'
        f'\t\t\t\t\t\t/YUnderline 1\n'
        f'\t\t\t\t\t\t/HindiNumbers false\n'
        f'\t\t\t\t\t\t/Kashida 1\n'
        f'\t\t\t\t\t>>\n'
        f'\t\t\t\t>>\n'
        f'\t\t\t>>\n'
        f'\t\t\t]\n'
        f'\t\t\t/RunLengthArray [ {tlen} ]\n'
        f'\t\t\t/IsJoinable 2\n'
        f'\t\t>>\n'
        f'\t\t/GridInfo\n'
        f'\t\t<<\n'
        f'\t\t\t/GridIsOn false\n'
        f'\t\t\t/ShowGrid false\n'
        f'\t\t\t/GridSize 18.0\n'
        f'\t\t\t/GridLeading 22.0\n'
        f'\t\t\t/GridColor\n'
        f'\t\t\t<<\n'
        f'\t\t\t\t/Type 1\n'
        f'\t\t\t\t/Values [ 0.0 0.0 0.0 1.0 ]\n'
        f'\t\t\t>>\n'
        f'\t\t\t/GridLeadingFillColor\n'
        f'\t\t\t<<\n'
        f'\t\t\t\t/Type 1\n'
        f'\t\t\t\t/Values [ 0.0 0.0 0.0 1.0 ]\n'
        f'\t\t\t>>\n'
        f'\t\t\t/AlignLineHeightToGridFlags false\n'
        f'\t\t>>\n'
        f'\t\t/AntiAlias 4\n'
        f'\t\t/UseFractionalGlyphWidths true\n'
        f'\t\t/RenderingIntent 0\n'
        f'\t>>\n'
        f'\t/ResourceDict\n'
        f'\t<<\n'
        f'\t\t/KinsokuSet [\n'
        f'\t\t<<\n'
        f'\t\t\t/Name ({kinsoku_esc})\n'
        f'\t\t\t/NoStart (\\xfe\\xff\\x30\\x01\\x30\\x02\\xff\\x0c\\xff\\x0e)\n'
        f'\t\t\t/NoEnd (\\xfe\\xff\\x20\\x18\\x20\\x1c\\xff\\x08\\x30\\x14)\n'
        f'\t\t\t/Keep (\\xfe\\xff\\x20\\x15\\x20\\x25)\n'
        f'\t\t\t/Hanging (\\xfe\\xff\\x30\\x01\\x30\\x02\\x00.\\x00,)\n'
        f'\t\t>>\n'
        f'\t\t]\n'
        f'\t\t/MojiKumiSet [\n'
        f'\t\t<<\n'
        f'\t\t\t/InternalName ({moji_esc})\n'
        f'\t\t>>\n'
        f'\t\t]\n'
        f'\t\t/TheNormalStyleSheet 0\n'
        f'\t\t/TheNormalParagraphSheet 0\n'
        f'\t\t/ParagraphSheetSet [\n'
        f'\t\t<<\n'
        f'\t\t\t/Name ({nrgb_esc})\n'
        f'\t\t\t/DefaultStyleSheet 0\n'
        f'\t\t\t/Properties\n'
        f'\t\t\t<<\n'
        f'\t\t\t\t/Justification 0\n'
        f'\t\t\t\t/FirstLineIndent 0.0\n'
        f'\t\t\t\t/StartIndent 0.0\n'
        f'\t\t\t\t/EndIndent 0.0\n'
        f'\t\t\t\t/SpaceBefore 0.0\n'
        f'\t\t\t\t/SpaceAfter 0.0\n'
        f'\t\t\t\t/AutoHyphenate true\n'
        f'\t\t\t\t/HyphenatedWordSize 6\n'
        f'\t\t\t\t/PreHyphen 2\n'
        f'\t\t\t\t/PostHyphen 2\n'
        f'\t\t\t\t/ConsecutiveHyphens 8\n'
        f'\t\t\t\t/Zone 36.0\n'
        f'\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]\n'
        f'\t\t\t\t/LetterSpacing [ 0.0 0.0 0.0 ]\n'
        f'\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]\n'
        f'\t\t\t\t/AutoLeading 1.2\n'
        f'\t\t\t\t/LeadingType 0\n'
        f'\t\t\t\t/Hanging false\n'
        f'\t\t\t\t/Burasagari false\n'
        f'\t\t\t\t/KinsokuOrder 0\n'
        f'\t\t\t\t/EveryLineComposer false\n'
        f'\t\t\t>>\n'
        f'\t\t>>\n'
        f'\t\t]\n'
        f'\t\t/StyleSheetSet [\n'
        f'\t\t<<\n'
        f'\t\t\t/Name ({nrgb_esc})\n'
        f'\t\t\t/StyleSheetData\n'
        f'\t\t\t<<\n'
        f'\t\t\t\t/Font 0\n'
        f'\t\t\t\t/FontSize 12.0\n'
        f'\t\t\t\t/FauxBold false\n'
        f'\t\t\t\t/FauxItalic false\n'
        f'\t\t\t\t/AutoLeading true\n'
        f'\t\t\t\t/Leading 0.0\n'
        f'\t\t\t\t/HorizontalScale 1.0\n'
        f'\t\t\t\t/VerticalScale 1.0\n'
        f'\t\t\t\t/Tracking 0\n'
        f'\t\t\t\t/AutoKerning true\n'
        f'\t\t\t\t/Kerning 0\n'
        f'\t\t\t\t/BaselineShift 0.0\n'
        f'\t\t\t\t/FontCaps 0\n'
        f'\t\t\t\t/FontBaseline 0\n'
        f'\t\t\t\t/Underline false\n'
        f'\t\t\t\t/Strikethrough false\n'
        f'\t\t\t\t/Ligatures true\n'
        f'\t\t\t\t/DLigatures false\n'
        f'\t\t\t\t/BaselineDirection 2\n'
        f'\t\t\t\t/Tsume 0.0\n'
        f'\t\t\t\t/StyleRunAlignment 2\n'
        f'\t\t\t\t/Language 0\n'
        f'\t\t\t\t/NoBreak false\n'
        f'\t\t\t\t/FillColor\n'
        f'\t\t\t\t<<\n'
        f'\t\t\t\t\t/Type 1\n'
        f'\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]\n'
        f'\t\t\t\t>>\n'
        f'\t\t\t\t/StrokeColor\n'
        f'\t\t\t\t<<\n'
        f'\t\t\t\t\t/Type 1\n'
        f'\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]\n'
        f'\t\t\t\t>>\n'
        f'\t\t\t\t/FillFlag true\n'
        f'\t\t\t\t/StrokeFlag false\n'
        f'\t\t\t\t/FillFirst true\n'
        f'\t\t\t\t/YUnderline 1\n'
        f'\t\t\t\t/OutlineWidth 1.0\n'
        f'\t\t\t\t/CharacterDirection 0\n'
        f'\t\t\t\t/HindiNumbers false\n'
        f'\t\t\t\t/Kashida 1\n'
        f'\t\t\t\t/DiacriticPos 2\n'
        f'\t\t\t>>\n'
        f'\t\t>>\n'
        f'\t\t]\n'
        f'\t\t/FontSet [\n'
        f'\t\t<<\n'
        f'\t\t\t/Name ({font_esc})\n'
        f'\t\t\t/Script 0\n'
        f'\t\t\t/FontType 0\n'
        f'\t\t\t/Synthetic 0\n'
        f'\t\t>>\n'
        f'\t\t<<\n'
        f'\t\t\t/Name ({fb_esc})\n'
        f'\t\t\t/Script 0\n'
        f'\t\t\t/FontType 0\n'
        f'\t\t\t/Synthetic 0\n'
        f'\t\t>>\n'
        f'\t\t]\n'
        f'\t\t/SuperscriptSize .583\n'
        f'\t\t/SuperscriptPosition .333\n'
        f'\t\t/SubscriptSize .583\n'
        f'\t\t/SubscriptPosition .333\n'
        f'\t\t/SmallCapSize .7\n'
        f'\t>>\n'
        f'\t/DocumentResources\n'
        f'\t<<\n'
        f'\t\t/KinsokuSet []\n'
        f'\t\t/MojiKumiSet []\n'
        f'\t\t/TheNormalStyleSheet 0\n'
        f'\t\t/TheNormalParagraphSheet 0\n'
        f'\t\t/ParagraphSheetSet []\n'
        f'\t\t/StyleSheetSet []\n'
        f'\t\t/FontSet []\n'
        f'\t\t/SuperscriptSize .583\n'
        f'\t\t/SuperscriptPosition .333\n'
        f'\t\t/SubscriptSize .583\n'
        f'\t\t/SubscriptPosition .333\n'
        f'\t\t/SmallCapSize .7\n'
        f'\t>>\n'
        f'>>\n'
    )
    return ed.encode('ascii')


def make_tysh_block(text, x, y, w, h, font_size=24.0,
                     r=1.0, g=1.0, b=1.0, font_name='ArialMT'):
    """Build TySh block using real Photoshop binary template."""
    buf = io.BytesIO()

    # Version + Transform (50 bytes)
    buf.write(pk('>H', 1))
    buf.write(pk('>d', 1.0))     # xx
    buf.write(pk('>d', 0.0))     # xy
    buf.write(pk('>d', 0.0))     # yx
    buf.write(pk('>d', 1.0))     # yy
    buf.write(pk('>d', float(x)))  # tx
    buf.write(pk('>d', float(y)))  # ty

    # Section B: descriptor header with text length updated
    text_with_null = text + '\x00'
    sec_b = _TYSH_SEC_B[:-4] + pk('>I', len(text_with_null))

    # Section C: text content in UTF-16BE
    sec_c = text_with_null.encode('utf-16-be')

    # Build EngineData
    engine_data = _build_engine_data(text, font_name, font_size, r, g, b)

    # Section D: update EngineData length (last 4 bytes)
    sec_d = _TYSH_SEC_D[:-4] + pk('>I', len(engine_data))

    # Section F: warp + final bounds
    warp_part = _TYSH_SEC_F[:-32]
    bounds = pk('>dddd', float(x), float(y), float(x + w), float(y + h))
    sec_f = warp_part + bounds

    buf.write(sec_b)
    buf.write(sec_c)
    buf.write(sec_d)
    buf.write(engine_data)
    buf.write(sec_f)

    tysh_data = buf.getvalue()
    block = b'8BIM' + b'TySh' + pk('>I', len(tysh_data)) + tysh_data
    if len(block) % 2:
        block += b'\x00'
    return block


def build_text_layer(name, text, x, y, w, h, font_size, W, H, lid,
                      r=1.0, g=1.0, b=1.0, font_name='ArialMT', opacity=255):
    """Build EDITABLE text layer with TySh block."""
    top = max(0, y)
    left = max(0, x)
    bottom = min(H, y + h)
    right = min(W, x + w)

    # 4 channels (NOT 5 — text layers don't have user mask channel)
    ch_ids = [-1, 0, 1, 2]
    ch_data_each = pk('>H', 0)

    rec = pk('>IIII', top, left, bottom, right)
    rec += pk('>H', 4)
    for ch_id in ch_ids:
        rec += pk('>hI', ch_id, len(ch_data_each))

    # flags=8 (not 24 — text layers are not adjustment layers)
    rec += b'8BIM' + b'norm' + pk('>BBBB', opacity, 0, 8, 0)

    extra = pk('>I', 0)  # mask data
    br = make_blending_ranges()
    extra += pk('>I', len(br)) + br
    extra += pstring(name, 4)
    extra += make_tysh_block(text, x, y, w, h, font_size, r, g, b, font_name)
    extra += make_common_extras(name, lid, is_adj=False)

    rec += pk('>I', len(extra)) + extra
    return rec, ch_data_each * 4


# =============================================================================
# Text detection — Uses Claude AI on Railway (replaces Google Vision)
# =============================================================================

def detect_text(image_bytes, railway_url):
    """
    Detect text in image using Claude AI on Railway server.
    Falls back to Google Vision if Railway URL not set.
    """
    if not image_bytes:
        print('[TextDetect] No image')
        return []

    # Method 1: Claude AI via Railway (PREFERRED)
    if railway_url:
        try:
            print(f'[TextDetect] Using Claude AI at {railway_url}/detect-text')
            resp = requests.post(
                f'{railway_url}/detect-text',
                files={'image': ('image.jpg', image_bytes, 'image/jpeg')},
                timeout=30
            )
            print(f'[TextDetect] Status: {resp.status_code}')

            if resp.status_code == 200:
                data = resp.json()
                texts = data.get('texts', [])
                print(f'[TextDetect] Claude found {len(texts)} texts')
                if texts:
                    print(f'[TextDetect] Full text: {data.get("full_text", "")[:100]}')
                return texts
            else:
                print(f'[TextDetect] Error: {resp.text[:300]}')
        except Exception as e:
            print(f'[TextDetect] Claude error: {e}')
            traceback.print_exc()

    # Method 2: Google Vision API (FALLBACK)
    api_key = GOOGLE_VISION_API_KEY
    if api_key:
        try:
            print('[TextDetect] Falling back to Google Vision')
            b64 = base64.b64encode(image_bytes).decode('utf-8')
            body = {"requests": [{"image": {"content": b64},
                                   "features": [{"type": "TEXT_DETECTION", "maxResults": 20}]}]}
            resp = requests.post(
                f'https://vision.googleapis.com/v1/images:annotate?key={api_key}',
                json=body, timeout=15)

            if resp.status_code == 200:
                data = resp.json()
                first = data.get('responses', [{}])[0]
                annotations = first.get('textAnnotations', [])
                texts = []
                for i, ann in enumerate(annotations):
                    if i == 0:
                        continue
                    verts = ann.get('boundingPoly', {}).get('vertices', [])
                    if len(verts) >= 4:
                        x = verts[0].get('x', 0)
                        y = verts[0].get('y', 0)
                        w = verts[1].get('x', 0) - x
                        h = verts[2].get('y', 0) - y
                        texts.append({'text': ann.get('description', ''),
                                      'x': x, 'y': y, 'w': max(w, 1), 'h': max(h, 1)})
                print(f'[TextDetect] Vision found {len(texts)} texts')
                return texts
        except Exception as e:
            print(f'[TextDetect] Vision error: {e}')

    print('[TextDetect] No detection method available')
    return []


# =============================================================================
# Vignette
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
# Layer builders
# =============================================================================

def build_pixel_layer(name, img, blend, opacity, W, H, lid):
    img_rgba = img.convert('RGBA').resize((W, H), Image.LANCZOS)
    arr = np.array(img_rgba, dtype=np.uint8)
    if 'Subject' not in name and 'Vignette' not in name and 'Text' not in name:
        arr[:, :, 3] = 255

    chs = [(-1, 3), (0, 0), (1, 1), (2, 2)]
    ch_parts = []
    for ch_id, ch_idx in chs:
        plane = arr[:, :, ch_idx]
        row_counts, compressed = rle_encode_channel(plane)
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
    ch_data_each = pk('>H', 0)

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


# =============================================================================
# PSD assembler
# =============================================================================

def create_psd(layer_specs, W, H, original_rgb):
    s1 = b'8BPS' + pk('>H', 1) + b'\x00' * 6
    s1 += pk('>H', 3) + pk('>I', H) + pk('>I', W)
    s1 += pk('>H', 8) + pk('>H', 3)

    s2 = pk('>I', 0)
    s3 = pk('>I', 0)

    all_records = b''
    all_chdata = b''
    for spec in layer_specs:
        if spec['type'] == 'pixel':
            rec, chd = build_pixel_layer(spec['name'], spec['image'], spec['blend_mode'],
                                          spec['opacity'], W, H, spec['lid'])
        elif spec['type'] == 'text':
            rec, chd = build_text_layer(
                spec['name'], spec['text'],
                spec['x'], spec['y'], spec['w'], spec['h'],
                spec.get('font_size', 24.0), W, H, spec['lid'],
                r=spec.get('r', 1.0), g=spec.get('g', 1.0), b=spec.get('b', 1.0),
                font_name=spec.get('font_name', 'ArialMT'),
                opacity=spec.get('opacity', 255))
        else:
            rec, chd = build_adjustment_layer(spec['name'], spec['adj_block'], spec['blend_mode'],
                                               spec['opacity'], W, H, spec['lid'])
        all_records += rec
        all_chdata += chd

    li = pk('>h', len(layer_specs)) + all_records + all_chdata
    if len(li) % 2:
        li += b'\x00'
    body = pk('>I', len(li)) + li + pk('>I', 0)
    s4 = pk('>I', len(body)) + body

    merged_arr = np.array(original_rgb, dtype=np.uint8)
    all_row_counts = []
    all_compressed = []
    for c in range(3):
        plane = merged_arr[:, :, c]
        row_counts, compressed = rle_encode_channel(plane)
        all_row_counts.extend(row_counts)
        all_compressed.append(compressed)

    s5 = pk('>H', 1)
    for rc in all_row_counts:
        s5 += pk('>H', rc)
    for comp in all_compressed:
        s5 += comp

    return s1 + s2 + s3 + s4 + s5


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
@app.route('/health')
def health():
    return jsonify({
        "status": "ok", "service": "LayerAI PSD Pro", "version": "2.1.0",
        "format": "ALL LEGACY",
        "layers": "brit ✓ curv ✓ hue2 ✓ levl ✓ blnc ✓",
        "vision": "on" if GOOGLE_VISION_API_KEY else "off",
        "removebg": "on" if REMOVE_BG_API_KEY else "off"
    })

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

        specs.append({'type': 'pixel', 'name': 'Background',
                      'image': orig.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1

        if REMOVE_BG_API_KEY:
            try:
                rsp = requests.post('https://api.remove.bg/v1.0/removebg',
                                    files={'image_file': ('i.jpg', raw, 'image/jpeg')},
                                    data={'size': 'auto'},
                                    headers={'X-Api-Key': REMOVE_BG_API_KEY}, timeout=20)
                if rsp.status_code == 200:
                    subj = Image.open(io.BytesIO(rsp.content)).convert('RGBA')
                    specs.append({'type': 'pixel', 'name': 'Subject Masked',
                                  'image': subj, 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
                    lid += 1
            except Exception as e:
                print('removebg:', e)

        specs.append({'type': 'adjustment', 'name': 'Curves 1',
                      'adj_block': make_curv_block(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1

        specs.append({'type': 'adjustment', 'name': 'Brightness/Contrast 1',
                      'adj_block': make_brit_block(20, 10), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1

        specs.append({'type': 'pixel', 'name': 'Vignette',
                      'image': make_vignette(W, H), 'blend_mode': 'mul ', 'opacity': 180, 'lid': lid})

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
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        brightness = int(request.form.get('brightness', 0))
        contrast = int(request.form.get('contrast', 0))
        hue = int(request.form.get('hue', 0))
        saturation = int(request.form.get('saturation', 0))
        lightness = int(request.form.get('lightness', 0))
        color_grade = request.form.get('color_grade', 'Cinematic')
        effects = request.form.get('effects', '')

        cb_shadow_cr = int(request.form.get('cb_shadow_cr', 0))
        cb_shadow_mg = int(request.form.get('cb_shadow_mg', 0))
        cb_shadow_yb = int(request.form.get('cb_shadow_yb', 0))
        cb_midtone_cr = int(request.form.get('cb_midtone_cr', 0))
        cb_midtone_mg = int(request.form.get('cb_midtone_mg', 0))
        cb_midtone_yb = int(request.form.get('cb_midtone_yb', 0))
        cb_highlight_cr = int(request.form.get('cb_highlight_cr', 0))
        cb_highlight_mg = int(request.form.get('cb_highlight_mg', 0))
        cb_highlight_yb = int(request.form.get('cb_highlight_yb', 0))

        lvl_shadows = int(request.form.get('lvl_shadows', 0))
        lvl_midtones = int(request.form.get('lvl_midtones', 100))
        lvl_highlights = int(request.form.get('lvl_highlights', 255))
        lvl_out_shadows = int(request.form.get('lvl_out_shadows', 0))
        lvl_out_highlights = int(request.form.get('lvl_out_highlights', 255))

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

        specs.append({'type': 'pixel', 'name': 'Background',
                      'image': orig.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1

        if REMOVE_BG_API_KEY:
            try:
                rsp = requests.post('https://api.remove.bg/v1.0/removebg',
                                    files={'image_file': ('i.jpg', raw, 'image/jpeg')},
                                    data={'size': 'auto'},
                                    headers={'X-Api-Key': REMOVE_BG_API_KEY}, timeout=20)
                if rsp.status_code == 200:
                    subj = Image.open(io.BytesIO(rsp.content)).convert('RGBA')
                    specs.append({'type': 'pixel', 'name': 'Subject Masked',
                                  'image': subj, 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
                    lid += 1
            except Exception as e:
                print('removebg:', e)

        specs.append({'type': 'adjustment', 'name': 'Curves 1',
                      'adj_block': make_curv_block(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1

        specs.append({'type': 'adjustment', 'name': 'Brightness/Contrast 1',
                      'adj_block': make_brit_block(brightness, contrast),
                      'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1

        specs.append({'type': 'adjustment', 'name': 'Hue/Saturation 1',
                      'adj_block': make_hue2_block(hue=hue, saturation=saturation, lightness=lightness),
                      'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1

        specs.append({'type': 'adjustment', 'name': 'Levels 1',
                      'adj_block': make_levl_block(shadows=lvl_shadows, midtones=lvl_midtones,
                                                    highlights=lvl_highlights,
                                                    output_shadows=lvl_out_shadows,
                                                    output_highlights=lvl_out_highlights),
                      'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1

        specs.append({'type': 'adjustment', 'name': 'Color Balance 1',
                      'adj_block': make_blnc_block(shadow_cr=cb_shadow_cr, shadow_mg=cb_shadow_mg,
                                                    shadow_yb=cb_shadow_yb,
                                                    midtone_cr=cb_midtone_cr, midtone_mg=cb_midtone_mg,
                                                    midtone_yb=cb_midtone_yb,
                                                    highlight_cr=cb_highlight_cr, highlight_mg=cb_highlight_mg,
                                                    highlight_yb=cb_highlight_yb),
                      'blend_mode': 'norm', 'opacity': 255, 'lid': lid})
        lid += 1

        if 'vignette' in effects.lower() or not effects:
            specs.append({'type': 'pixel', 'name': 'Vignette',
                          'image': make_vignette(W, H), 'blend_mode': 'mul ', 'opacity': 180, 'lid': lid})
            lid += 1

        if color_grade:
            if 'warm' in color_grade.lower():
                grade_img = Image.new('RGBA', (W, H), (255, 180, 120, 25))
            elif 'cool' in color_grade.lower():
                grade_img = Image.new('RGBA', (W, H), (120, 180, 255, 25))
            else:
                grade_img = Image.new('RGBA', (W, H), (200, 200, 180, 20))
            specs.append({'type': 'pixel', 'name': 'Color Grade - ' + color_grade,
                          'image': grade_img, 'blend_mode': 'over', 'opacity': 60, 'lid': lid})
            lid += 1

        if RAILWAY_API_URL or GOOGLE_VISION_API_KEY:
            # Resize image for text detection if > 4MB (Claude API 5MB limit)
            detect_bytes = raw
            if len(raw) > 4 * 1024 * 1024:
                detect_img = Image.open(io.BytesIO(raw))
                detect_img.thumbnail((1024, 1024), Image.LANCZOS)
                detect_buf = io.BytesIO()
                detect_img.save(detect_buf, format='JPEG', quality=85)
                detect_bytes = detect_buf.getvalue()
                print(f'[TextDetect] Resized {len(raw)} -> {len(detect_bytes)} bytes')

            texts = detect_text(detect_bytes, RAILWAY_API_URL)
            for t in texts[:10]:
                scale_x = W / orig_W if orig_W != W else 1
                scale_y = H / orig_H if orig_H != H else 1
                tx = int(t['x'] * scale_x)
                ty = int(t['y'] * scale_y)
                tw = max(int(t.get('w', 100) * scale_x), 20)
                th = max(int(t.get('h', 30) * scale_y), 15)
                font_size = max(12.0, min(72.0, th * 0.8))
                specs.append({
                    'type': 'text',
                    'name': 'Text: ' + t['text'][:20],
                    'text': t['text'],
                    'x': tx, 'y': ty, 'w': tw, 'h': th,
                    'font_size': font_size,
                    'blend_mode': 'norm', 'opacity': 255, 'lid': lid
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

# === Test/Debug Endpoints ===

@app.route('/test-vision', methods=['POST'])
def test_vision():
    result = {
        "railway_url_set": bool(RAILWAY_API_URL),
        "google_vision_set": bool(GOOGLE_VISION_API_KEY),
        "method": "claude_ai" if RAILWAY_API_URL else ("google_vision" if GOOGLE_VISION_API_KEY else "none")
    }
    if not RAILWAY_API_URL and not GOOGLE_VISION_API_KEY:
        result["error"] = "Neither RAILWAY_API_URL nor GOOGLE_VISION_API_KEY set"
        return jsonify(result), 500
    if 'image' not in request.files:
        result["error"] = "Send image as multipart form with key 'image'"
        return jsonify(result), 400

    raw = request.files['image'].read()
    result["image_size"] = len(raw)
    try:
        img = Image.open(io.BytesIO(raw))
        result["dimensions"] = f"{img.size[0]}x{img.size[1]}"
    except:
        result["error"] = "Invalid image"
        return jsonify(result), 400

    texts = detect_text(raw, RAILWAY_API_URL)
    result["texts"] = texts
    result["count"] = len(texts)
    result["status"] = "ok" if texts else "no_text_found"
    return jsonify(result)

@app.route('/test-adjustments', methods=['GET'])
def test_adjustments():
    """Test PSD with non-zero values — verify correct display in Photoshop."""
    W, H = 400, 300
    bg = Image.new('RGBA', (W, H), (40, 40, 50, 255))
    draw = ImageDraw.Draw(bg)
    for y in range(H):
        draw.line([(0, y), (W, y)], fill=(int(80 + 100 * y / H), int(60 + 80 * y / H), int(40 + 120 * y / H), 255))

    specs, lid = [], 1
    specs.append({'type': 'pixel', 'name': 'Background', 'image': bg.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Curves 1', 'adj_block': make_curv_block(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Brightness/Contrast 1', 'adj_block': make_brit_block(15, 10), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Hue/Saturation 1', 'adj_block': make_hue2_block(hue=10, saturation=-15, lightness=5), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Levels 1', 'adj_block': make_levl_block(shadows=10, midtones=120, highlights=245), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Color Balance 1', 'adj_block': make_blnc_block(midtone_cr=8, midtone_mg=-3, midtone_yb=-10), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

    psd = create_psd(specs, W, H, bg.convert('RGB'))
    buf = io.BytesIO(psd); buf.seek(0)
    return send_file(buf, mimetype='application/octet-stream', as_attachment=True, download_name='test_v21.psd')

@app.route('/test-zero', methods=['GET'])
def test_zero():
    """Test PSD with ALL ZERO values — image should look unchanged."""
    W, H = 400, 300
    bg = Image.new('RGBA', (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(bg)
    for y in range(H):
        for x in range(0, W, 4):
            draw.rectangle([x, y, x + 3, y], fill=(int(255 * x / W), int(255 * y / H), int(255 * (1 - x / W)), 255))

    specs, lid = [], 1
    specs.append({'type': 'pixel', 'name': 'Background', 'image': bg.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Curves 1', 'adj_block': make_curv_block(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Brightness/Contrast 1', 'adj_block': make_brit_block(0, 0), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Hue/Saturation 1', 'adj_block': make_hue2_block(0, 0, 0), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Levels 1', 'adj_block': make_levl_block(0, 100, 255), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
    specs.append({'type': 'adjustment', 'name': 'Color Balance 1', 'adj_block': make_blnc_block(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

    psd = create_psd(specs, W, H, bg.convert('RGB'))
    buf = io.BytesIO(psd); buf.seek(0)
    return send_file(buf, mimetype='application/octet-stream', as_attachment=True, download_name='test_zero.psd')

@app.route('/test-single-layer', methods=['GET'])
def test_single_layer():
    t = request.args.get('type', 'hue2')
    W, H = 200, 200
    bg = Image.new('RGBA', (W, H), (128, 128, 128, 255))

    m = {
        'hue2': ('Hue/Saturation 1', make_hue2_block(hue=20, saturation=-30, lightness=5)),
        'levl': ('Levels 1', make_levl_block(shadows=15, midtones=130, highlights=240)),
        'blnc': ('Color Balance 1', make_blnc_block(midtone_cr=15, midtone_yb=-20)),
        'brit': ('Brightness/Contrast 1', make_brit_block(25, 15)),
        'curv': ('Curves 1', make_curv_block()),
    }
    if t not in m:
        return jsonify({"error": f"Unknown: {t}", "valid": list(m.keys())}), 400

    name, block = m[t]
    specs = [
        {'type': 'pixel', 'name': 'Background', 'image': bg, 'blend_mode': 'norm', 'opacity': 255, 'lid': 1},
        {'type': 'adjustment', 'name': name, 'adj_block': block, 'blend_mode': 'norm', 'opacity': 255, 'lid': 2}
    ]
    psd = create_psd(specs, W, H, bg.convert('RGB'))
    buf = io.BytesIO(psd); buf.seek(0)
    return send_file(buf, mimetype='application/octet-stream', as_attachment=True, download_name=f'test_{t}.psd')

@app.route('/verify-psd', methods=['POST'])
def verify_psd():
    if 'file' not in request.files:
        return jsonify({"error": "Upload with key 'file'"}), 400
    data = request.files['file'].read()
    results = {"file_size": len(data), "valid": data[:4] == b'8BPS', "blocks": {}}
    for key in [b'hue2', b'levl', b'blnc', b'brit', b'curv']:
        pos = data.find(key)
        if pos >= 0:
            blen = struct.unpack('>I', data[pos + 4:pos + 8])[0] if pos + 8 <= len(data) else -1
            results["blocks"][key.decode()] = {"offset": f"0x{pos:04x}", "length": blen}
    return jsonify(results)

@app.route('/debug-env', methods=['GET'])
def debug_env():
    return jsonify({
        "REMOVE_BG_API_KEY": "SET" if REMOVE_BG_API_KEY else "NOT SET",
        "GOOGLE_VISION_API_KEY": "SET" if GOOGLE_VISION_API_KEY else "NOT SET",
    })

if __name__ == '__main__':
    print("=" * 50)
    print("  LayerAI PSD Pro — V2.1.0 FINAL")
    print("  ALL LEGACY FORMAT")
    print("  brit ✓ curv ✓ hue2 ✓ levl ✓ blnc ✓")
    print("=" * 50)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
