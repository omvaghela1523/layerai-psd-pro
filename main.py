"""
LayerAI PSD Pro — V5.0 FINAL
==============================
Production-ready PSD generator matching reference PSD structure.

Deploy: Render (Python Flask)
Env vars needed:
  - REMOVE_BG_API_KEY    → Subject extraction (remove.bg)
  - STABILITY_API_KEY    → Background inpainting (stability.ai)
  - RAILWAY_API_URL      → Claude AI analysis (your Railway server URL)

NO Google Vision needed — text detection done by Claude on Railway.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests as http_requests
import io, os, struct, base64, json, time, traceback
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Environment variables ────────────────────────────────────────────────────
REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
RAILWAY_API_URL   = os.environ.get("RAILWAY_API_URL", "")

# ── Helper: struct pack shortcut ─────────────────────────────────────────────
def pk(fmt, *a):
    return struct.pack(fmt, *a)

def pstring(s, pad=4):
    b = s.encode('ascii', errors='replace')[:255]
    raw = bytes([len(b)]) + b
    while len(raw) % pad:
        raw += b'\x00'
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# RLE COMPRESSION — optimized with memoryview
# ═══════════════════════════════════════════════════════════════════════════════

def rle_encode_row(row_bytes):
    out = bytearray()
    mv = memoryview(row_bytes)
    i, n = 0, len(mv)
    while i < n:
        if i + 1 < n and mv[i] == mv[i + 1]:
            val = mv[i]
            run = 2
            limit = min(n, i + 128)
            while i + run < limit and mv[i + run] == val:
                run += 1
            out.append((257 - run) & 0xFF)
            out.append(val)
            i += run
        else:
            j = i + 1
            limit = min(n, i + 128)
            while j < limit:
                if j + 1 < n and mv[j] == mv[j + 1]:
                    break
                j += 1
            out.append(j - i - 1)
            out.extend(mv[i:j])
            i = j
    return bytes(out)

def rle_encode_channel(plane_2d):
    H = plane_2d.shape[0]
    row_data, row_counts = [], []
    for y in range(H):
        enc = rle_encode_row(plane_2d[y].tobytes())
        row_counts.append(len(enc))
        row_data.append(enc)
    return row_counts, b''.join(row_data)

def rle_pack_channel(plane_2d):
    H = plane_2d.shape[0]
    row_counts, compressed = rle_encode_channel(plane_2d)
    parts = [pk('>H', 1)]
    parts.extend(pk('>H', rc) for rc in row_counts)
    parts.append(compressed)
    return b''.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL LAYER DATA BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def make_additional(key, data):
    block = b'8BIM' + key + pk('>I', len(data)) + data
    if len(block) % 2:
        block += b'\x00'
    return block

def make_luni(name):
    return make_additional(b'luni', pk('>I', len(name)) + name.encode('utf-16-be') + b'\x00\x00')

def make_lnsr(t):    return make_additional(b'lnsr', t)
def make_lyid(lid):  return make_additional(b'lyid', pk('>I', lid))
def make_clbl():     return make_additional(b'clbl', b'\x01\x00\x00\x00')
def make_infx():     return make_additional(b'infx', b'\x00\x00\x00\x00')
def make_knko():     return make_additional(b'knko', b'\x00\x00\x00\x00')
def make_lspf():     return make_additional(b'lspf', b'\x00\x00\x00\x00')
def make_lclr():     return make_additional(b'lclr', b'\x00' * 8)
def make_fxrp():     return make_additional(b'fxrp', b'\x00' * 16)

def make_shmd():
    d = bytes.fromhex("000000013842494d63757374000000000000003400000010000000010000000000086d6574616461746100000001000000096c6179657254696d6564")
    return make_additional(b'shmd', d)

def make_blending_ranges():
    return b''.join(pk('>HH', 0, 65535) for _ in range(10))

def make_adj_mask_data():
    return pk('>iiii', 0, 0, 0, 0) + pk('>BB', 255, 0) + b'\x00\x00'

def make_common_extras(name, lid, is_adj=False, is_text=False):
    e = make_luni(name)
    e += make_lnsr(b'cont' if is_adj else b'rend' if is_text else b'layr')
    e += make_lyid(lid)
    e += make_clbl() + make_infx() + make_knko() + make_lspf() + make_lclr()
    e += make_shmd() + make_fxrp()
    return e


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE RESOURCES — 300 PPI
# ═══════════════════════════════════════════════════════════════════════════════

def make_image_resources(ppi=300):
    rd = pk('>I', ppi * 65536) + pk('>HH', 1, 2) + pk('>I', ppi * 65536) + pk('>HH', 1, 2)
    return b'8BIM' + pk('>H', 1005) + b'\x00\x00' + pk('>I', len(rd)) + rd


# ═══════════════════════════════════════════════════════════════════════════════
# ADJUSTMENT LAYER BLOCKS — ALL LEGACY FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

def make_brit_block(brightness=0, contrast=0):
    return make_additional(b'brit', pk('>hh', brightness, contrast) + pk('>h', 128) + pk('>B', 0) + b'\x00')

def make_curv_block(points=None):
    if points is None:
        points = [(0, 0), (255, 255)]
    npts = len(points)
    data = bytearray()
    data += pk('>HH', 0, 1)
    data += pk('>H', 0)  # composite: default
    data += pk('>H', npts)
    for out_v, in_v in points:
        data += pk('>H', out_v) + pk('>H', in_v)
    data += b'Crv '
    data += pk('>H', 4) + pk('>I', 1) + pk('>H', npts)
    for out_v, in_v in points:
        data += pk('>H', out_v) + pk('>H', in_v)
    if len(data) % 2:
        data += b'\x00'
    return make_additional(b'curv', bytes(data))

def make_hue2_block(hue=0, saturation=0, lightness=0):
    data = pk('>H', 2) + pk('>B', 0) + b'\x00'
    data += pk('>hhhh', 0, 0, 0, 0) + pk('>hhh', hue, saturation, lightness)
    for r1, r2, r3, r4 in [(315,345,15,45),(15,45,75,105),(75,105,135,165),
                            (135,165,195,225),(195,225,255,285),(255,285,315,345)]:
        data += pk('>hhhh', r1, r2, r3, r4) + pk('>hhh', 0, 0, 0)
    return make_additional(b'hue2', data)

def make_levl_block(shadows=0, midtones=100, highlights=255, out_shadows=0, out_highlights=255):
    data = pk('>H', 2)
    data += pk('>HHHHH', shadows, highlights, out_shadows, out_highlights, midtones)
    for _ in range(28):
        data += pk('>HHHHH', 0, 255, 0, 255, 100)
    return make_additional(b'levl', data)

def make_blnc_block(shadow_cr=0, shadow_mg=0, shadow_yb=0,
                    midtone_cr=0, midtone_mg=0, midtone_yb=0,
                    highlight_cr=0, highlight_mg=0, highlight_yb=0):
    data = pk('>hhh', shadow_cr, shadow_mg, shadow_yb)
    data += pk('>hhh', midtone_cr, midtone_mg, midtone_yb)
    data += pk('>hhh', highlight_cr, highlight_mg, highlight_yb)
    data += pk('>B', 1) + b'\x00'
    return make_additional(b'blnc', data)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER EFFECTS — lrFX (Bevel & Emboss + Stroke) from reference PSD
# ═══════════════════════════════════════════════════════════════════════════════

_LRFX_DATA = bytes.fromhex("000000073842494d636d6e5300000007000000000100003842494d6473647700000033000000020000000000000000001200000019000000006d6e8f8fb4b300003842494d6e6f726d0000ff00006d6e8f8fb4b300003842494d697364770000003300000002009e000000000000005a0000000000000000f8f7e20d9c3700003842494d6d756c200001ff0000f8f7e20d9c3700003842494d6f676c770000002a0000000200040000000000000000ffffffffffff00003842494d7363726e001a0000ffffffffffff00003842494d69676c770000002b0000000200000000000000000000fffffffff1f100003842494d6e6f726d006b010000fffffffff1f100003842494d6265766c0000004e00000002ff4d0000000570a3000400003842494d6c6464673842494d6d756c200000ffffffffffff00000000000000000000000002b5540100000000ffffffffffff0000000000000000000000003842494d736f666900000022000000023842494d6e6f726d0000191a2324090a0000ff000000191a2324090a00000000")

def make_lrfx_block():
    return make_additional(b'lrFX', _LRFX_DATA)


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT LAYER — TySh (editable text with EngineData)
# ═══════════════════════════════════════════════════════════════════════════════

def _utf16be_escape(text_bytes):
    return ''.join(chr(b) if 32 <= b < 127 and b not in (40,41,92) else f'\\x{b:02x}' for b in text_bytes)

def _build_engine_data(text, font_name='ArialMT', font_size=24.0, r=1.0, g=1.0, b=1.0):
    text_u16 = b'\xfe\xff' + text.encode('utf-16-be') + b'\x00\x0d'
    text_esc = _utf16be_escape(text_u16)
    font_esc = _utf16be_escape(b'\xfe\xff' + font_name.encode('utf-16-be'))
    fb_esc = _utf16be_escape(b'\xfe\xff' + 'MyriadPro-Regular'.encode('utf-16-be'))
    nrgb_esc = _utf16be_escape(b'\xfe\xff' + 'Normal RGB'.encode('utf-16-be'))
    tlen = len(text) + 1

    ed = f"""

<<
\t/EngineDict
\t<<
\t\t/Editor
\t\t<<
\t\t\t/Text ({text_esc})
\t\t>>
\t\t/ParagraphRun
\t\t<<
\t\t\t/DefaultRunData
\t\t\t<<
\t\t\t\t/ParagraphSheet
\t\t\t\t<<
\t\t\t\t\t/DefaultStyleSheet 0
\t\t\t\t\t/Properties
\t\t\t\t\t<<
\t\t\t\t\t>>
\t\t\t\t>>
\t\t\t\t/Adjustments
\t\t\t\t<<
\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]
\t\t\t\t\t/XY [ 0.0 0.0 ]
\t\t\t\t>>
\t\t\t>>
\t\t\t/RunArray [
\t\t\t<<
\t\t\t\t/ParagraphSheet
\t\t\t\t<<
\t\t\t\t\t/DefaultStyleSheet 0
\t\t\t\t\t/Properties
\t\t\t\t\t<<
\t\t\t\t\t\t/Justification 0
\t\t\t\t\t\t/FirstLineIndent 0.0
\t\t\t\t\t\t/StartIndent 0.0
\t\t\t\t\t\t/EndIndent 0.0
\t\t\t\t\t\t/SpaceBefore 0.0
\t\t\t\t\t\t/SpaceAfter 0.0
\t\t\t\t\t\t/AutoHyphenate false
\t\t\t\t\t\t/HyphenatedWordSize 6
\t\t\t\t\t\t/PreHyphen 2
\t\t\t\t\t\t/PostHyphen 2
\t\t\t\t\t\t/ConsecutiveHyphens 8
\t\t\t\t\t\t/Zone 36.0
\t\t\t\t\t\t/WordSpacing [ .8 1.0 1.33 ]
\t\t\t\t\t\t/LetterSpacing [ 0.0 0.0 0.0 ]
\t\t\t\t\t\t/GlyphSpacing [ 1.0 1.0 1.0 ]
\t\t\t\t\t\t/AutoLeading 1.2
\t\t\t\t\t\t/LeadingType 0
\t\t\t\t\t\t/Hanging false
\t\t\t\t\t\t/Burasagari false
\t\t\t\t\t\t/KinsokuOrder 0
\t\t\t\t\t\t/EveryLineComposer false
\t\t\t\t\t>>
\t\t\t\t>>
\t\t\t\t/Adjustments
\t\t\t\t<<
\t\t\t\t\t/Axis [ 1.0 0.0 1.0 ]
\t\t\t\t\t/XY [ 0.0 0.0 ]
\t\t\t\t>>
\t\t\t>>
\t\t\t]
\t\t\t/RunLengthArray [ {tlen} ]
\t\t\t/IsJoinable 1
\t\t>>
\t\t/StyleRun
\t\t<<
\t\t\t/DefaultRunData
\t\t\t<<
\t\t\t\t/StyleSheet
\t\t\t\t<<
\t\t\t\t\t/StyleSheetData
\t\t\t\t\t<<
\t\t\t\t\t>>
\t\t\t\t>>
\t\t\t>>
\t\t\t/RunArray [
\t\t\t<<
\t\t\t\t/StyleSheet
\t\t\t\t<<
\t\t\t\t\t/StyleSheetData
\t\t\t\t\t<<
\t\t\t\t\t\t/Font 0
\t\t\t\t\t\t/FontSize {font_size:.1f}
\t\t\t\t\t\t/FauxBold false
\t\t\t\t\t\t/FauxItalic false
\t\t\t\t\t\t/AutoLeading true
\t\t\t\t\t\t/Leading .01
\t\t\t\t\t\t/HorizontalScale 1.0
\t\t\t\t\t\t/VerticalScale 1.0
\t\t\t\t\t\t/Tracking 0
\t\t\t\t\t\t/AutoKerning true
\t\t\t\t\t\t/BaselineShift 0.0
\t\t\t\t\t\t/FontCaps 0
\t\t\t\t\t\t/FontBaseline 0
\t\t\t\t\t\t/Underline false
\t\t\t\t\t\t/Strikethrough false
\t\t\t\t\t\t/Ligatures true
\t\t\t\t\t\t/BaselineDirection 1
\t\t\t\t\t\t/Tsume 0.0
\t\t\t\t\t\t/StyleRunAlignment 2
\t\t\t\t\t\t/Language 0
\t\t\t\t\t\t/NoBreak false
\t\t\t\t\t\t/FillColor
\t\t\t\t\t\t<<
\t\t\t\t\t\t\t/Type 1
\t\t\t\t\t\t\t/Values [ 1.0 {r:.4f} {g:.4f} {b:.4f} ]
\t\t\t\t\t\t>>
\t\t\t\t\t\t/StrokeColor
\t\t\t\t\t\t<<
\t\t\t\t\t\t\t/Type 1
\t\t\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]
\t\t\t\t\t\t>>
\t\t\t\t\t>>
\t\t\t\t>>
\t\t\t>>
\t\t\t]
\t\t\t/RunLengthArray [ {tlen} ]
\t\t\t/IsJoinable 2
\t\t>>
\t\t/GridInfo
\t\t<<
\t\t\t/GridIsOn false
\t\t\t/ShowGrid false
\t\t\t/GridSize 18.0
\t\t\t/GridLeading 22.0
\t\t\t/GridColor
\t\t\t<<
\t\t\t\t/Type 1
\t\t\t\t/Values [ 0.0 0.0 0.0 1.0 ]
\t\t\t>>
\t\t\t/GridLeadingFillColor
\t\t\t<<
\t\t\t\t/Type 1
\t\t\t\t/Values [ 0.0 0.0 0.0 1.0 ]
\t\t\t>>
\t\t\t/AlignLineHeightToGridFlags false
\t\t>>
\t\t/AntiAlias 4
\t\t/UseFractionalGlyphWidths true
\t\t/RenderingIntent 0
\t>>
\t/ResourceDict
\t<<
\t\t/KinsokuSet [
\t\t<<
\t\t\t/Name (\\xfe\\xff\\x00P\\x00h\\x00o\\x00t\\x00o\\x00s\\x00h\\x00o\\x00p\\x00K\\x00i\\x00n\\x00s\\x00o\\x00k\\x00u\\x00H\\x00a\\x00r\\x00d)
\t\t\t/NoStart (\\xfe\\xff)
\t\t\t/NoEnd (\\xfe\\xff)
\t\t\t/Keep (\\xfe\\xff)
\t\t\t/Hanging (\\xfe\\xff)
\t\t>>
\t\t]
\t\t/MojiKumiSet [
\t\t<<
\t\t\t/InternalName (\\xfe\\xff\\x00d\\x00e\\x00f\\x00a\\x00u\\x00l\\x00t)
\t\t>>
\t\t]
\t\t/TheNormalStyleSheet 0
\t\t/TheNormalParagraphSheet 0
\t\t/ParagraphSheetSet [
\t\t<<
\t\t\t/Name ({nrgb_esc})
\t\t\t/DefaultStyleSheet 0
\t\t\t/Properties
\t\t\t<<
\t\t\t>>
\t\t>>
\t\t]
\t\t/StyleSheetSet [
\t\t<<
\t\t\t/Name ({nrgb_esc})
\t\t\t/StyleSheetData
\t\t\t<<
\t\t\t\t/Font 0
\t\t\t\t/FontSize 12.0
\t\t\t\t/FauxBold false
\t\t\t\t/FauxItalic false
\t\t\t\t/AutoLeading true
\t\t\t\t/Kerning 0
\t\t\t\t/BaselineShift 0.0
\t\t\t\t/Tracking 0
\t\t\t\t/FillColor
\t\t\t\t<<
\t\t\t\t\t/Type 1
\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]
\t\t\t\t>>
\t\t\t\t/StrokeColor
\t\t\t\t<<
\t\t\t\t\t/Type 1
\t\t\t\t\t/Values [ 1.0 0.0 0.0 0.0 ]
\t\t\t\t>>
\t\t\t>>
\t\t>>
\t\t]
\t\t/FontSet [
\t\t<<
\t\t\t/Name ({font_esc})
\t\t\t/Script 0
\t\t\t/FontType 0
\t\t\t/Synthetic 0
\t\t>>
\t\t<<
\t\t\t/Name ({fb_esc})
\t\t\t/Script 0
\t\t\t/FontType 0
\t\t\t/Synthetic 0
\t\t>>
\t\t]
\t\t/SuperscriptSize .583
\t\t/SuperscriptPosition .333
\t\t/SubscriptSize .583
\t\t/SubscriptPosition .333
\t\t/SmallCapSize .7
\t>>
\t/DocumentResources
\t<<
\t\t/KinsokuSet []
\t\t/MojiKumiSet []
\t\t/TheNormalStyleSheet 0
\t\t/TheNormalParagraphSheet 0
\t\t/ParagraphSheetSet []
\t\t/StyleSheetSet []
\t\t/FontSet []
\t\t/SuperscriptSize .583
\t\t/SuperscriptPosition .333
\t\t/SubscriptSize .583
\t\t/SubscriptPosition .333
\t\t/SmallCapSize .7
\t>>
>>
"""
    return ed.encode('ascii')


def make_tysh_block(text, x, y, w, h, font_size=24.0, r=1.0, g=1.0, b=1.0, font_name='ArialMT'):
    buf = io.BytesIO()
    buf.write(pk('>H', 1))
    for v in [1.0, 0.0, 0.0, 1.0, float(x), float(y)]:
        buf.write(pk('>d', v))
    # Minimal descriptor header
    buf.write(pk('>H', 50))  # text descriptor version
    buf.write(pk('>I', 0))   # classID name length
    buf.write(b'\x00\x00\x00\x00TxLr')
    buf.write(pk('>I', 1))   # descriptor count
    buf.write(pk('>I', 0))   # key name length
    buf.write(b'Txt ')
    buf.write(b'TEXT')
    text_u16 = text.encode('utf-16-be')
    buf.write(pk('>I', len(text) + 1))
    buf.write(text_u16 + b'\x00\x00')
    # EngineData
    engine_data = _build_engine_data(text, font_name, font_size, r, g, b)
    buf.write(pk('>I', 0))  # key name length
    buf.write(b'textGridding')
    buf.write(b'enum')
    buf.write(pk('>I', 0))
    buf.write(b'textGridding')
    buf.write(pk('>I', 0))
    buf.write(b'None')
    buf.write(pk('>I', 0))
    buf.write(b'Ornt')
    buf.write(b'enum')
    buf.write(pk('>I', 0))
    buf.write(b'Ornt')
    buf.write(pk('>I', 0))
    buf.write(b'Hrzn')
    buf.write(pk('>I', 0))
    buf.write(b'AntA')
    buf.write(b'enum')
    buf.write(pk('>I', 0))
    buf.write(b'Annt')
    buf.write(pk('>I', 14))
    buf.write(b'antiAliasSharp')
    buf.write(pk('>I', 0))
    buf.write(b'bounds')
    buf.write(b'Objc')
    buf.write(pk('>I', 1) + pk('>I', 0) + pk('>I', 0))
    buf.write(b'bounds')
    buf.write(pk('>I', 4))
    for kn, val in [('Left', float(x)), ('Top ', float(y)), ('Rght', float(x+w)), ('Btom', float(y+h))]:
        buf.write(pk('>I', 0))
        buf.write(kn.encode('ascii'))
        buf.write(b'UntF')
        buf.write(b'#Pnt')
        buf.write(pk('>d', val))
    buf.write(pk('>I', 0))
    buf.write(b'boundingBox')
    buf.write(b'Objc')
    buf.write(pk('>I', 1) + pk('>I', 0) + pk('>I', 0))
    buf.write(b'boundingBox')
    buf.write(pk('>I', 4))
    for kn, val in [('Left', float(x)), ('Top ', float(y)), ('Rght', float(x+w)), ('Btom', float(y+h))]:
        buf.write(pk('>I', 0))
        buf.write(kn.encode('ascii'))
        buf.write(b'UntF')
        buf.write(b'#Pnt')
        buf.write(pk('>d', val))
    buf.write(pk('>I', 0))
    buf.write(b'TextIndex')
    buf.write(b'long')
    buf.write(pk('>I', 1))
    buf.write(pk('>I', 0))
    buf.write(b'EngineData')
    buf.write(b'tdta')
    buf.write(pk('>I', len(engine_data)))
    buf.write(engine_data)
    # Warp
    buf.write(pk('>HI', 1, 0) + pk('>I', 0))
    buf.write(b'warp')
    buf.write(pk('>I', 5))
    buf.write(pk('>I', 0) + b'warpStyle' + b'enum' + pk('>I', 0) + b'warpStyle' + pk('>I', 0) + b'warpNone')
    buf.write(pk('>I', 0) + b'warpValue' + b'doub' + pk('>d', 0.0))
    buf.write(pk('>I', 0) + b'warpPerspective' + b'doub' + pk('>d', 0.0))
    buf.write(pk('>I', 0) + b'warpPerspectiveOther' + b'doub' + pk('>d', 0.0))
    buf.write(pk('>I', 0) + b'warpRotate' + b'enum' + pk('>I', 0) + b'Ornt' + pk('>I', 0) + b'Hrzn')
    # Bounds
    buf.write(pk('>dddd', float(x), float(y), float(x+w), float(y+h)))
    
    tysh_data = buf.getvalue()
    block = b'8BIM' + b'TySh' + pk('>I', len(tysh_data)) + tysh_data
    if len(block) % 2:
        block += b'\x00'
    return block


# ═══════════════════════════════════════════════════════════════════════════════
# SOFT SHADOW GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def make_soft_shadow(subject_rgba, offset_x=10, offset_y=30, blur_radius=25, opacity=140):
    W, H = subject_rgba.size
    alpha = subject_rgba.split()[-1]
    shadow = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    black = Image.new('RGBA', (W, H), (0, 0, 0, opacity))
    shadow.paste(black, (0, 0), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
    if offset_x or offset_y:
        off = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        off.paste(shadow, (offset_x, offset_y))
        shadow = off
    return shadow


# ═══════════════════════════════════════════════════════════════════════════════
# INPAINTING — Stability AI (clean background generation)
# ═══════════════════════════════════════════════════════════════════════════════

def inpaint_background(original_bytes, subject_mask, prompt="natural background, photorealistic"):
    if not STABILITY_API_KEY:
        print('[Inpaint] STABILITY_API_KEY not set')
        return None
    try:
        orig = Image.open(io.BytesIO(original_bytes)).convert('RGB')
        W, H = orig.size
        scale = 1.0
        if W * H > 1024 * 1024:
            scale = (1024 * 1024 / (W * H)) ** 0.5
            orig = orig.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
            subject_mask = subject_mask.resize((int(W*scale), int(H*scale)), Image.LANCZOS)

        img_buf = io.BytesIO(); orig.save(img_buf, format='PNG'); img_buf.seek(0)
        mask_dilated = subject_mask.filter(ImageFilter.MaxFilter(15))
        mask_buf = io.BytesIO(); mask_dilated.convert('L').save(mask_buf, format='PNG'); mask_buf.seek(0)

        resp = http_requests.post(
            "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
            headers={"authorization": f"Bearer {STABILITY_API_KEY}", "accept": "image/*"},
            files={"image": ("image.png", img_buf, "image/png"), "mask": ("mask.png", mask_buf, "image/png")},
            data={"prompt": prompt, "output_format": "png", "mode": "mask"},
            timeout=60
        )
        if resp.status_code != 200:
            print(f'[Inpaint] Error {resp.status_code}: {resp.text[:200]}')
            return None
        result = Image.open(io.BytesIO(resp.content)).convert('RGB')
        if scale != 1.0:
            result = result.resize((W, H), Image.LANCZOS)
        print(f'[Inpaint] Success')
        return result
    except Exception as e:
        print(f'[Inpaint] {e}')
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT DETECTION (via Claude on Railway — no Google Vision)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_text(image_bytes):
    if not RAILWAY_API_URL or not image_bytes:
        return []
    try:
        resp = http_requests.post(
            f'{RAILWAY_API_URL}/detect-text',
            files={'image': ('image.jpg', image_bytes, 'image/jpeg')},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            texts = data.get('texts', [])
            print(f'[TextDetect] Found {len(texts)} texts')
            return texts
    except Exception as e:
        print(f'[TextDetect] {e}')
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_pixel_layer(name, img, blend, opacity, W, H, lid):
    is_overlay = any(k in name for k in ('Subject', 'Shadow', 'Vignette'))
    if is_overlay:
        img_rgba = img.convert('RGBA')
    else:
        img_rgba = img.convert('RGBA').resize((W, H), Image.LANCZOS)
        arr_tmp = np.array(img_rgba, dtype=np.uint8)
        arr_tmp[:, :, 3] = 255
        img_rgba = Image.fromarray(arr_tmp)

    lw, lh = img_rgba.size
    arr = np.array(img_rgba, dtype=np.uint8)

    ch_packed = []
    for ch_id, ch_idx in [(-1,3),(0,0),(1,1),(2,2)]:
        ch_packed.append((ch_id, rle_pack_channel(arr[:, :, ch_idx])))

    parts = [pk('>iiii', 0, 0, lh, lw), pk('>H', 4)]
    for ch_id, packed in ch_packed:
        parts.append(pk('>hI', ch_id, len(packed)))

    bm = blend.encode('ascii').ljust(4)[:4]
    parts.append(b'8BIM' + bm + pk('>BBBB', opacity, 0, 8, 0))

    br = make_blending_ranges()
    extra = b''.join([pk('>I', 0), pk('>I', len(br)), br, pstring(name, 4),
                      make_common_extras(name, lid)])
    parts.append(pk('>I', len(extra)) + extra)

    return b''.join(parts), b''.join(p for _, p in ch_packed)


def build_text_layer(name, text, x, y, w, h, font_size, W, H, lid,
                     r=1.0, g=1.0, b=1.0, font_name='ArialMT', opacity=255):
    top, left = max(0, y), max(0, x)
    bottom, right = min(H, y+h), min(W, x+w)
    ch_data_each = pk('>H', 0)

    parts = [pk('>iiii', top, left, bottom, right), pk('>H', 4)]
    for ch_id in [-1, 0, 1, 2]:
        parts.append(pk('>hI', ch_id, len(ch_data_each)))

    parts.append(b'8BIM' + b'norm' + pk('>BBBB', opacity, 0, 0x28, 0))

    extra_parts = [pk('>I', 0)]
    br = make_blending_ranges()
    extra_parts.append(pk('>I', len(br)) + br)
    extra_parts.append(pstring(name, 4))
    extra_parts.append(make_lrfx_block())
    extra_parts.append(make_tysh_block(text, x, y, w, h, font_size, r, g, b, font_name))
    extra_parts.append(make_common_extras(name, lid, is_text=True))
    extra = b''.join(extra_parts)
    parts.append(pk('>I', len(extra)) + extra)

    return b''.join(parts), ch_data_each * 4


def build_adjustment_layer(name, adj_block, blend, opacity, W, H, lid):
    ch_data_each = pk('>H', 0)
    parts = [pk('>iiii', 0, 0, 0, 0), pk('>H', 5)]
    for ch_id in [-1, 0, 1, 2, -2]:
        parts.append(pk('>hI', ch_id, len(ch_data_each)))

    bm = blend.encode('ascii').ljust(4)[:4]
    parts.append(b'8BIM' + bm + pk('>BBBB', opacity, 0, 0x18, 0))

    mask = make_adj_mask_data()
    br = make_blending_ranges()
    extra = pk('>I', len(mask)) + mask + pk('>I', len(br)) + br + pstring(name, 4)
    extra += adj_block + make_common_extras(name, lid, is_adj=True)
    parts.append(pk('>I', len(extra)) + extra)

    return b''.join(parts), ch_data_each * 5


# ═══════════════════════════════════════════════════════════════════════════════
# PSD ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

def create_psd(layer_specs, W, H, original_rgb, ppi=300):
    t0 = time.time()

    s1 = b'8BPS' + pk('>H', 1) + b'\x00' * 6 + pk('>H', 4) + pk('>I', H) + pk('>I', W) + pk('>H', 8) + pk('>H', 3)
    s2 = pk('>I', 0)
    s3_data = make_image_resources(ppi)
    s3 = pk('>I', len(s3_data)) + s3_data

    rec_parts, chd_parts = [], []
    for spec in layer_specs:
        t = spec['type']
        if t == 'pixel':
            rec, chd = build_pixel_layer(spec['name'], spec['image'], spec['blend_mode'], spec['opacity'], W, H, spec['lid'])
        elif t == 'text':
            rec, chd = build_text_layer(spec['name'], spec['text'], spec['x'], spec['y'], spec['w'], spec['h'],
                                        spec.get('font_size', 24.0), W, H, spec['lid'],
                                        spec.get('r', 1.0), spec.get('g', 1.0), spec.get('b', 1.0),
                                        spec.get('font_name', 'ArialMT'), spec.get('opacity', 255))
        else:
            rec, chd = build_adjustment_layer(spec['name'], spec['adj_block'], spec['blend_mode'], spec['opacity'], W, H, spec['lid'])
        rec_parts.append(rec)
        chd_parts.append(chd)

    t1 = time.time()
    li = pk('>h', -len(layer_specs)) + b''.join(rec_parts) + b''.join(chd_parts)
    if len(li) % 2: li += b'\x00'
    body = pk('>I', len(li)) + li + pk('>I', 0)
    s4 = pk('>I', len(body)) + body

    merged = np.array(original_rgb, dtype=np.uint8)
    alpha_plane = np.full((H, W), 255, dtype=np.uint8)
    s5p = [pk('>H', 1)]
    all_rc, all_comp = [], []
    for c in range(3):
        rc, comp = rle_encode_channel(merged[:, :, c])
        all_rc.extend(rc); all_comp.append(comp)
    rc, comp = rle_encode_channel(alpha_plane)
    all_rc.extend(rc); all_comp.append(comp)
    s5p.extend(pk('>H', r) for r in all_rc)
    s5p.extend(all_comp)

    t2 = time.time()
    print(f'[PSD] {t2-t0:.2f}s total ({len(layer_specs)} layers)')
    return s1 + s2 + s3 + s4 + b''.join(s5p)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
@app.route('/health')
def health():
    return jsonify({
        "status": "ok", "service": "LayerAI PSD Pro", "version": "5.0.0",
        "apis": {
            "remove_bg": "on" if REMOVE_BG_API_KEY else "OFF",
            "stability_ai": "on" if STABILITY_API_KEY else "OFF",
            "railway_claude": "on" if RAILWAY_API_URL else "OFF"
        }
    })


@app.route('/generate-psd-pro', methods=['POST'])
def gen_psd_pro():
    """
    FULL 7-STEP PIPELINE:
    1. Extract subject (Remove.bg)
    2. Inpaint clean background (Stability AI)
    3. Background as bottom layer
    4. Subject as masked layer
    5. Soft shadow under subject
    6. Color adjustments (from Claude's analysis)
    7. Clean layer naming
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Read params (from Claude analysis or defaults)
        brightness = int(request.form.get('brightness', 0))
        contrast = int(request.form.get('contrast', 0))
        hue = int(request.form.get('hue', 0))
        saturation = int(request.form.get('saturation', 0))
        lightness = int(request.form.get('lightness', 0))
        lvl_shadows = int(request.form.get('lvl_shadows', 0))
        lvl_midtones = int(request.form.get('lvl_midtones', 100))
        lvl_highlights = int(request.form.get('lvl_highlights', 255))
        cb_midtone_cr = int(request.form.get('cb_midtone_cr', 0))
        cb_midtone_mg = int(request.form.get('cb_midtone_mg', 0))
        cb_midtone_yb = int(request.form.get('cb_midtone_yb', 0))
        shadow_offset_x = int(request.form.get('shadow_offset_x', 10))
        shadow_offset_y = int(request.form.get('shadow_offset_y', 30))
        shadow_blur = int(request.form.get('shadow_blur', 25))
        shadow_opacity = int(request.form.get('shadow_opacity', 140))
        inpaint_prompt = request.form.get('inpaint_prompt', 'natural background, photorealistic, clean, no people')

        raw = request.files['image'].read()
        orig = Image.open(io.BytesIO(raw)).convert('RGBA')
        W, H = orig.size
        print(f'[Pipeline] {W}x{H}')

        specs, lid = [], 1

        # STEP 1: Extract subject
        subject_rgba = None
        if REMOVE_BG_API_KEY:
            try:
                rsp = http_requests.post('https://api.remove.bg/v1.0/removebg',
                    files={'image_file': ('i.jpg', raw, 'image/jpeg')},
                    data={'size': 'auto'}, headers={'X-Api-Key': REMOVE_BG_API_KEY}, timeout=30)
                if rsp.status_code == 200:
                    subject_rgba = Image.open(io.BytesIO(rsp.content)).convert('RGBA')
                    if subject_rgba.size != (W, H):
                        subject_rgba = subject_rgba.resize((W, H), Image.LANCZOS)
                    print('[Step 1] Subject extracted')
            except Exception as e:
                print(f'[Step 1] {e}')

        # STEP 2: Inpaint background
        bg_image = orig.convert('RGB')
        if subject_rgba and STABILITY_API_KEY:
            mask = subject_rgba.split()[-1]
            inpainted = inpaint_background(raw, mask, inpaint_prompt)
            if inpainted:
                bg_image = inpainted
                print('[Step 2] Background inpainted')

        # STEP 3: Background layer
        specs.append({'type': 'pixel', 'name': 'Backgroundd', 'image': bg_image.convert('RGBA'),
                      'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

        # Text detection
        texts = detect_text(raw)
        for t in texts[:5]:
            tx, ty = int(t.get('x', 50)), int(t.get('y', 50))
            tw, th = max(int(t.get('w', 100)), 20), max(int(t.get('h', 30)), 15)
            specs.append({'type': 'text', 'name': t['text'][:20], 'text': t['text'],
                          'x': tx, 'y': ty, 'w': tw, 'h': th, 'font_size': max(12.0, th * 0.8),
                          'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

        # STEP 4: Subject layer
        if subject_rgba:
            specs.append({'type': 'pixel', 'name': 'Subject', 'image': subject_rgba,
                          'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

            # STEP 5: Shadow
            shadow = make_soft_shadow(subject_rgba, shadow_offset_x, shadow_offset_y, shadow_blur, shadow_opacity)
            specs.append({'type': 'pixel', 'name': 'Shadow', 'image': shadow,
                          'blend_mode': 'mul ', 'opacity': 200, 'lid': lid}); lid += 1

        # STEP 6: Adjustments
        specs.append({'type': 'adjustment', 'name': 'Brightness/Contrast 1',
                      'adj_block': make_brit_block(brightness, contrast),
                      'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

        specs.append({'type': 'adjustment', 'name': 'Levels 1',
                      'adj_block': make_levl_block(lvl_shadows, lvl_midtones, lvl_highlights),
                      'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

        specs.append({'type': 'adjustment', 'name': 'Curves 1',
                      'adj_block': make_curv_block([(0,0),(87,93),(255,255)]),
                      'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

        if hue or saturation or lightness:
            specs.append({'type': 'adjustment', 'name': 'Hue/Saturation 1',
                          'adj_block': make_hue2_block(hue, saturation, lightness),
                          'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

        if cb_midtone_cr or cb_midtone_mg or cb_midtone_yb:
            specs.append({'type': 'adjustment', 'name': 'Color Balance 1',
                          'adj_block': make_blnc_block(midtone_cr=cb_midtone_cr, midtone_mg=cb_midtone_mg, midtone_yb=cb_midtone_yb),
                          'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

        psd = create_psd(specs, W, H, bg_image if isinstance(bg_image, Image.Image) and bg_image.mode == 'RGB' else bg_image.convert('RGB') if hasattr(bg_image, 'convert') else orig.convert('RGB'), ppi=300)
        buf = io.BytesIO(psd); buf.seek(0)
        return send_file(buf, mimetype='application/octet-stream', as_attachment=True, download_name='layerai-pro.psd')
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/generate-psd', methods=['POST'])
def gen_psd_simple():
    """Simple PSD — no inpainting, just layers."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image"}), 400
        raw = request.files['image'].read()
        orig = Image.open(io.BytesIO(raw)).convert('RGBA')
        W, H = orig.size

        specs, lid = [], 1
        specs.append({'type': 'pixel', 'name': 'Backgroundd', 'image': orig.copy(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

        if REMOVE_BG_API_KEY:
            try:
                rsp = http_requests.post('https://api.remove.bg/v1.0/removebg',
                    files={'image_file': ('i.jpg', raw, 'image/jpeg')}, data={'size': 'auto'},
                    headers={'X-Api-Key': REMOVE_BG_API_KEY}, timeout=20)
                if rsp.status_code == 200:
                    subj = Image.open(io.BytesIO(rsp.content)).convert('RGBA')
                    if subj.size != (W, H): subj = subj.resize((W, H), Image.LANCZOS)
                    specs.append({'type': 'pixel', 'name': 'Subject', 'image': subj, 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
            except: pass

        specs.append({'type': 'adjustment', 'name': 'Brightness/Contrast 1', 'adj_block': make_brit_block(0, 0), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
        specs.append({'type': 'adjustment', 'name': 'Levels 1', 'adj_block': make_levl_block(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1
        specs.append({'type': 'adjustment', 'name': 'Curves 1', 'adj_block': make_curv_block(), 'blend_mode': 'norm', 'opacity': 255, 'lid': lid}); lid += 1

        psd = create_psd(specs, W, H, orig.convert('RGB'), ppi=300)
        buf = io.BytesIO(psd); buf.seek(0)
        return send_file(buf, mimetype='application/octet-stream', as_attachment=True, download_name='layerai-export.psd')
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/debug-env')
def debug_env():
    return jsonify({
        "REMOVE_BG_API_KEY": "SET" if REMOVE_BG_API_KEY else "NOT SET",
        "STABILITY_API_KEY": "SET" if STABILITY_API_KEY else "NOT SET",
        "RAILWAY_API_URL": RAILWAY_API_URL or "NOT SET"
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  LayerAI PSD Pro — V5.0.0 FINAL")
    print("  No Google Vision — Claude text detection only")
    print("  APIs: Remove.bg + Stability AI + Claude")
    print("=" * 50)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
