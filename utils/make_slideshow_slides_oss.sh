#!/usr/bin/env bash
###############################################################################
# make_slideshow_slides.sh — create a narrated slideshow video
#
# Dependencies
# • ffmpeg & ffprobe
# • whisper CLI
# • Python 3 with: pillow, pysrt, requests, numpy, scikit-learn, jieba,
#                  networkx, tqdm
# • Ollama running a local model (default gpt-oss:latest) on http://localhost:11434
#
# Usage:
#   ./make_slideshow_slides.sh BASENAME [CHUNK] [TITLE] [IMG_DIR] [LANG] [SPEED]
###############################################################################
set -euo pipefail

# ── CLI ---------------------------------------------------------------------
BASE="$1"                         # e.g. "tesla_q2_2025"
CHUNK="${2:-30}"                  # seconds per transcript slice
TITLE="${3:-$(echo "$BASE" | tr '_-' ' ')}"
IMG_DIR="${4:-$BASE}"             # folder containing images
LANG="${5:-English}"              # Whisper language
SPEED="${6:-1.05}"                # audio speed-up 0.5–2.0

# ── Layout ------------------------------------------------------------------
W=1920 ; H=1080
TITLE_SIZE=96
BODY_SIZE_EN=50                   # English bullets
BODY_SIZE_ZH=60                   # Chinese bullets
LEFT_PAD=220 ; RIGHT_PAD=80
LINE_SP=34  ; TOP_MARGIN=100
BG="#000000" ; FG="#FFFFFF"

# ── Ollama ------------------------------------------------------------------
: "${OLLAMA_URL:=http://localhost:11434/api/generate}"
: "${OLLAMA_MODEL:=gpt-oss:latest}"

LANG_LC=$(echo "$LANG" | tr '[:upper:]' '[:lower:]')
IS_ZH=$([[ "$LANG_LC" =~ ^(chinese|zh|zh_cn|zh-cn)$ ]] && echo 1 || echo 0)
BODY_SIZE=$([ "$IS_ZH" -eq 1 ] && echo "$BODY_SIZE_ZH" || echo "$BODY_SIZE_EN")

# ── Paths -------------------------------------------------------------------
if   [[ -f "$BASE.wav" ]]; then AUDIO_INPUT="$BASE.wav"
elif [[ -f "$BASE.m4a" ]]; then AUDIO_INPUT="$BASE.m4a"
else echo "❌  Neither $BASE.wav nor $BASE.m4a found" >&2 ; exit 1 ; fi

POD="$BASE.podcast.wav"
SRT_FILE="${POD%.*}.srt"
SLIDES_DIR="${BASE}_slides"
OUT_MP4="${BASE}_slides.mp4"

# Clean previous artifacts (including last run's slides folder)
if [[ -d "$SLIDES_DIR" ]]; then
  echo "🧹 Cleaning previous slides folder: $SLIDES_DIR"
  rm -rf "$SLIDES_DIR"
fi
rm -f "$POD" "$SRT_FILE" "$OUT_MP4"

# ── 1. Speed-up audio -------------------------------------------------------
ffmpeg -y -i "$AUDIO_INPUT" -filter:a atempo="$SPEED" \
       -ac 2 -ar 44100 -sample_fmt s16 "$POD"

# ── 2. Transcribe -----------------------------------------------------------
whisper "$POD" --model small --language "$LANG" --device cpu \
        --fp16 False --task transcribe --output_format srt -o .

# Recreate a fresh slides directory
mkdir -p "$SLIDES_DIR"

# ── 3. Generate PNG frames --------------------------------------------------
export W H TITLE_SIZE BODY_SIZE LEFT_PAD RIGHT_PAD LINE_SP TOP_MARGIN \
       BG FG TITLE SLIDES_DIR IMG_DIR IS_ZH SRT_FILE CHUNK \
       OLLAMA_URL OLLAMA_MODEL

python <<'PY'
import os, re, json, requests, pysrt, numpy as np, tqdm
import jieba, jieba.analyse, networkx as nx
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# ── env ---------------------------------------------------------------------
W,H        = int(os.environ["W"]), int(os.environ["H"])
TS         = int(os.environ["TITLE_SIZE"])
BS         = int(os.environ["BODY_SIZE"])
LEFT,RIGHT = int(os.environ["LEFT_PAD"]), int(os.environ["RIGHT_PAD"])
SP,TOP     = int(os.environ["LINE_SP"]), int(os.environ["TOP_MARGIN"])
BG,FG      = os.environ["BG"], os.environ["FG"]
TITLE      = os.environ["TITLE"]
SLIDES     = Path(os.environ["SLIDES_DIR"]); IMG_DIR=Path(os.environ["IMG_DIR"])
IS_ZH      = bool(int(os.environ["IS_ZH"]))
SRT_FILE   = os.environ["SRT_FILE"]; CHUNK=int(os.environ["CHUNK"])
OLLAMA_URL = os.environ["OLLAMA_URL"]; OLLAMA_MODEL=os.environ["OLLAMA_MODEL"]

# ── 3-a. Slice transcript ---------------------------------------------------
subs=pysrt.open(SRT_FILE)
blocks={}
for s in subs:
    idx=s.start.ordinal//1000//CHUNK
    blocks.setdefault(idx,[]).append(s.text.replace('\n',' '))

# ── 3-b. Summarisation helpers ---------------------------------------------
FIL_ZH=re.compile(r'(嗯+|啊+|呃+|哼+|呐+|呗+|啰+|哦+|噢+|吗+|然后|就是|那个|这个|其实|那么)')
FIL_EN_L=re.compile(r'^(and|but|so|because|well|okay|alright|anyway)\s+',re.I)
FIL_EN_I=re.compile(r'\b(?:so|well|really|basically|literally|just|you know|uh|um|uh-huh|like|sort of|kind of|kinda)\b',re.I)

def call_ollama(prompt:str)->str:
    try:
        r=requests.post(OLLAMA_URL,json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False},timeout=60)
        r.raise_for_status()
        return r.json().get("response","").strip()
    except Exception as e:
        print("⚠️  Ollama call failed:",e); return ""

def en_bullets(txt:str):
    resp=call_ollama("Give me 3-4 concise professional bullet points (≤15 words each) that summarise:\n\n"+txt)
    parts=re.split(r'(?:^\s*[\-\*•]\s+|\n)',resp,flags=re.M) if resp else []
    if len(parts)<2: parts=re.split(r'(?<=[.!?])\s+',resp)
    out=[]
    for p in parts:
        p=FIL_EN_I.sub("",p); p=FIL_EN_L.sub("",p).strip().capitalize()
        p=re.sub(r'\s-\s',' – ',p)        # English: normalize hyphen to en-dash
        if len(p.split())>=3 and p not in out:
            out.append(p if p.endswith(('.','!','?')) else p+'.')
        if len(out)==4: break
    return out

def _normalize_zh_dashes(s:str)->str:
    # Chinese: unify -, – , — , − to "——" (no spaces)
    s = re.sub(r'\s*[-–—−]\s*', '——', s)
    s = s.replace('———', '——')
    return s

def zh_bullets(txt:str):
    resp=call_ollama("请用专业简体中文将下列内容概括成 3-4 个要点，每点 ≤25 字：\n\n"+txt)
    out=[s.strip("•- \n") for s in re.split(r'[；\n]',resp) if s.strip()] if resp else []
    out=[_normalize_zh_dashes(FIL_ZH.sub("",s)) for s in out]
    return out[:4]

def textrank_cn(txt,k=4):
    txt=FIL_ZH.sub("",txt)
    sents=[s for s in re.split(r'[。！？]',txt) if s.strip()]
    if not sents: return []
    M=(TfidfVectorizer().fit_transform(sents)*TfidfVectorizer().fit_transform(sents).T).toarray()
    np.fill_diagonal(M,0); pr=nx.pagerank(nx.from_numpy_array(M))
    pick=[sents[i] for i in sorted(pr,key=pr.get,reverse=True)[:k]]
    pick=[s if s.endswith("。") else s+"。" for s in pick]
    return [_normalize_zh_dashes(s) for s in pick]

def bullets(txt:str):
    return (zh_bullets(txt) or textrank_cn(txt)) if IS_ZH else en_bullets(txt)

# ── 3-c. Font helper (language-aware) ---------------------------------------
def font_ok(sz,bold=False):
    if IS_ZH:
        order=[
            "/Library/Fonts/NotoSansSC-Regular.otf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    else:
        order=[
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/NotoSansSC-Regular.otf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ]
        if bold:
            order.insert(2,"/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    for p in order:
        try:
            f=ImageFont.truetype(p,sz)
            probe = "测——-" if IS_ZH else "A-–—"
            if ImageDraw.Draw(Image.new("RGB",(1,1))).textbbox((0,0),probe,font=f):
                return f
        except Exception:
            continue
    return ImageFont.load_default()

ft_title,ft_body=font_ok(TS,True),font_ok(BS)
tmp=ImageDraw.Draw(Image.new("RGB",(1,1)))
txtw=lambda t,f: tmp.textbbox((0,0),t,font=f)[2]
WRAP_MAX=W-LEFT-RIGHT-40
def wrap(t):
    toks=list(t) if IS_ZH else t.split()
    buf=""; lines=[]
    for tok in toks:
        trial=(buf+tok) if IS_ZH else f"{buf} {tok}".strip()
        if txtw(trial,ft_body)<=WRAP_MAX: buf=trial
        else: lines.append(buf); buf=tok
    if buf: lines.append(buf)
    return lines

# ── 3-d. Images (cycled / round-robin) -------------------------------------
def natural_key(p): return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)',p.stem)]
pics=sorted([p for p in Path(os.environ["IMG_DIR"]).glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'}],key=natural_key)

# Optional splash shown once if present
splash=next((p for p in pics if re.search(r'(thumbnail|封面)',p.stem,re.I)),None)
if splash: pics.remove(splash)

# Prepare round-robin index for images
have_pics = len(pics) > 0
img_idx = 0

# ── 3-e. Save helper --------------------------------------------------------
frame=0
def save(img): global frame; img.save(SLIDES/f"frame_{frame:06d}.png"); frame+=1

# splash frame ---------------------------------------------------------------
if splash:
    sp=Image.open(splash).convert("RGB"); sp.thumbnail((W,H))
    bg=Image.new("RGB",(W,H),BG); bg.paste(sp,((W-sp.width)//2,(H-sp.height)//2)); save(bg)

# ── 3-f. Build slides (bullet → image; images loop) ------------------------
for idx in tqdm.tqdm(sorted(blocks)):
    text=" ".join(blocks[idx])[:2000]

    # bullet slide
    slide=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(slide)
    d.text(((W-txtw(TITLE,ft_title))//2,TOP),TITLE,font=ft_title,fill=FG)
    y=TOP+TS+70
    for b in bullets(text):
        first=True
        for ln in wrap(b):
            prefix=("• " if not IS_ZH else "•") if first else "  "
            d.text((LEFT,y),prefix+ln,font=ft_body,fill=FG)
            y+=BS+SP; first=False
        y+=SP
        if y>H-120: break
    save(slide)

    # image slide — pick next picture in round-robin if any exist
    if have_pics:
        p = pics[img_idx]
        img_idx = (img_idx + 1) % len(pics)
        im=Image.open(p).convert("RGB"); im.thumbnail((W,H))
        bg=Image.new("RGB",(W,H),BG); bg.paste(im,((W-im.width)//2,(H-im.height)//2)); save(bg)

# No leftover image dump — images are looped during narration.

(SLIDES/"frames_count.txt").write_text(str(frame))
PY

# ── 4. Assemble video -------------------------------------------------------
DUR=$(ffprobe -i "$POD" -show_entries format=duration -v quiet -of csv=p=0)
FRAMES=$(<"$SLIDES_DIR/frames_count.txt")
FPS=$(python - <<EOF
print(float($FRAMES)/float($DUR))
EOF
)
ffmpeg -y -framerate "$FPS" -pattern_type glob -i "$SLIDES_DIR/frame_*.png" \
       -i "$POD" -c:v libx264 -r 30 -pix_fmt yuv420p -preset veryfast -crf 18 \
       -c:a aac -b:a 192k -shortest "$OUT_MP4"

echo "✅  $OUT_MP4 created — cleaned old slides dir; images loop round-robin; length matches audio"
###############################################################################
