import base64
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Aura: The Fashion Oracle", page_icon="🔮", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Assets that are part of the theme, not part of the collage
RESERVED_ASSETS = ("sfondo", "zebrato", "c097e1", "c09401", "usericon", "boticon", "readme")

MAX_COLLAGE_IMAGES = 10
COLLAGE_THUMB_SIZE = (500, 700)     # collage images are decorative; no need for full res
KMEANS_SAMPLE_SIZE = 20_000         # cap pixels fed to KMeans so it stays fast
MIN_PIXELS_PER_ITEM = 100
NUM_COLORS = 3


def get_api_key():
    """Read the OpenAI key from Streamlit secrets first, then the environment.

    st.secrets raises if no secrets file exists at all, so it is guarded.
    """
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# ASSET LOADING  (cached: these read from disk and must not rerun every time)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def file_to_base64(path, thumbnail=None):
    """Return a base64 JPEG string for an image on disk, optionally downscaled."""
    if thumbnail is None:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img = Image.open(path).convert("RGB")
    img.thumbnail(thumbnail)
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def list_images():
    """Split the images sitting next to app.py into theme assets and collage art."""
    theme, collage = {}, []
    try:
        for filename in sorted(os.listdir(ASSETS_DIR)):
            lower = filename.lower()
            if not lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            path = os.path.join(ASSETS_DIR, filename)
            matched = next((kw for kw in RESERVED_ASSETS if kw in lower), None)
            if matched:
                theme[matched] = path
            else:
                collage.append(path)
    except OSError:
        pass
    return theme, collage


def first_existing(theme, keywords):
    for kw in keywords:
        if kw in theme:
            return theme[kw]
    return None


theme_assets, collage_paths = list_images()

bg_path = first_existing(theme_assets, ["c097e1", "sfondo", "glitter"])
zebra_path = first_existing(theme_assets, ["c09401", "zebra", "zebrato"])
user_icon_path = first_existing(theme_assets, ["usericon"])
bot_icon_path = first_existing(theme_assets, ["boticon"])

user_avatar = Image.open(user_icon_path) if user_icon_path else "👤"
ai_avatar = Image.open(bot_icon_path) if bot_icon_path else "🐱"

# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------

if bg_path:
    bg_css_rule = f"""
    background-image: url("data:image/jpeg;base64,{file_to_base64(bg_path)}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    """
else:
    bg_css_rule = "background-color: #210038;"

if zebra_path:
    zebra_css_rule = f"""
    background-image: url("data:image/jpeg;base64,{file_to_base64(zebra_path)}") !important;
    background-size: cover !important;
    background-position: center !important;
    """
else:
    zebra_css_rule = (
        "background: repeating-linear-gradient(45deg, #000 0px, #000 8px, "
        "#FFF 8px, #FFF 16px);"
    )

html_collage_images = "".join(
    f'<img src="data:image/jpeg;base64,{file_to_base64(p, COLLAGE_THUMB_SIZE)}" '
    f'class="collage-img img-{i + 1}">'
    for i, p in enumerate(collage_paths[:MAX_COLLAGE_IMAGES])
)

st.markdown(f"""
<style>
    @import url('https://fonts.cdnfonts.com/css/chopin-script');
    @import url('https://fonts.cdnfonts.com/css/new-romantics');

    .stApp {{
        {bg_css_rule}
        overflow-x: hidden;
    }}

    html, body, [class*="css"],
    h1, h2, h3, h4, h5, h6, p, span, label, div, input, textarea, button, .stMarkdown {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-weight: 700 !important;
        text-transform: lowercase !important;
        color: #FFFFFF !important;
    }}

    .editorial-collage {{
        position: relative;
        width: 100vw;
        left: calc(-50vw + 50%);
        height: 70vh;
        min-height: 500px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-top: -3rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        box-shadow: 0 20px 50px rgba(0,0,0,0.8);
    }}

    .collage-img {{
        position: absolute;
        box-shadow: 0 15px 35px rgba(0,0,0,0.8);
        border: 1px solid #444;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        object-fit: cover;
        opacity: 0.85;
        filter: grayscale(20%) contrast(110%);
    }}

    .collage-img:hover {{
        transform: scale(1.15) rotate(0deg) !important;
        z-index: 100 !important;
        opacity: 1;
        box-shadow: 0 20px 50px rgba(255,255,255,0.4);
        filter: grayscale(0%) contrast(100%);
    }}

    .img-1  {{ top: -5%;  left: -2%;  width: 20%; transform: rotate(-8deg);  z-index: 1; }}
    .img-2  {{ bottom: -5%; left: 2%; width: 18%; transform: rotate(12deg);  z-index: 3; }}
    .img-3  {{ top: 35%;  left: -5%;  width: 19%; transform: rotate(-4deg);  z-index: 2; }}
    .img-7  {{ top: -10%; left: 15%;  width: 16%; transform: rotate(5deg);   z-index: 4; }}
    .img-10 {{ bottom: 5%; left: 18%; width: 15%; transform: rotate(-10deg); z-index: 2; }}
    .img-4  {{ top: -5%;  right: -2%; width: 22%; transform: rotate(7deg);   z-index: 1; }}
    .img-5  {{ bottom: -10%; right: 2%; width: 19%; transform: rotate(-10deg); z-index: 3; }}
    .img-6  {{ top: 35%;  right: -5%; width: 20%; transform: rotate(15deg);  z-index: 2; }}
    .img-8  {{ bottom: -5%; right: 18%; width: 17%; transform: rotate(-6deg); z-index: 4; }}
    .img-9  {{ top: 5%;   right: 15%; width: 16%; transform: rotate(-12deg); z-index: 5; }}

    .title-container {{
        z-index: 50;
        pointer-events: none;
        text-align: center;
        background: radial-gradient(circle, rgba(33,0,56,0.8) 0%, rgba(33,0,56,0.4) 40%, transparent 70%);
        padding: 60px;
        border-radius: 50%;
    }}

    .main-title {{
        font-family: 'Chopin Script', cursive !important;
        font-size: 10rem !important;
        text-align: center;
        margin-bottom: -20px;
        font-weight: normal !important;
        line-height: 1;
        {zebra_css_rule}
        color: transparent !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        filter: drop-shadow(0px 0px 20px rgba(255, 255, 255, 0.8));
        text-transform: none !important;
    }}

    .editorial-subtitle {{
        text-align: center;
        font-weight: 700 !important;
        letter-spacing: 5px;
        text-transform: lowercase !important;
        font-size: 1.5rem !important;
        margin-top: 10px;
        color: #FFFFFF !important;
        filter: drop-shadow(0px 2px 5px rgba(0,0,0,1));
    }}

    .chopin-silver-title {{
        font-family: 'Chopin Script', cursive !important;
        font-size: 4rem !important;
        text-align: center;
        font-weight: normal !important;
        color: #C0C0C0 !important;
        text-transform: none !important;
        filter: drop-shadow(0px 0px 10px rgba(192, 192, 192, 0.6));
        margin-top: 3rem;
        margin-bottom: 2rem;
        line-height: 1.2;
    }}

    .zebra-divider {{
        height: 10px;
        background: repeating-linear-gradient(45deg, #000, #000 15px, #C0C0C0 15px, #C0C0C0 30px);
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.4);
        border-radius: 5px;
        margin-top: 3rem; margin-bottom: 3rem;
    }}

    .stFileUploader > div > div {{
        background-color: rgba(0, 0, 0, 0.7) !important;
        border: 2px solid #C0C0C0 !important;
        border-radius: 0px !important;
    }}

    [data-testid="stChatInput"] {{
        background-color: rgba(0, 0, 0, 0.8) !important;
        border: 2px solid #C0C0C0 !important;
        border-radius: 0px !important;
    }}

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div {{
        font-family: 'New Romantics', 'Helvetica Neue', sans-serif !important;
        text-transform: lowercase !important;
        letter-spacing: 3px !important;
    }}

    [data-testid="stChatInput"] textarea {{
        color: #FFFFFF !important;
        font-weight: normal !important;
    }}

    [data-testid="stChatMessage"] img {{
        border-radius: 50% !important;
        object-fit: cover !important;
        background-color: transparent !important;
        mix-blend-mode: screen !important;
    }}

    .palette-container {{ display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin-top: 20px; }}
    .color-card {{
        background-color: #000;
        border-radius: 0px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.9);
        overflow: hidden; width: 140px; text-align: center; transition: transform 0.3s ease;
        border: 1px solid #C0C0C0;
    }}
    .color-card:hover {{ transform: translateY(-10px); box-shadow: 0 15px 25px rgba(192, 192, 192, 0.6); }}
    .color-box {{ height: 120px; width: 100%; }}
    .color-info {{ padding: 15px 5px; }}
    .hex-code {{ font-weight: 700; color: #FFFFFF !important; font-size: 16px; letter-spacing: 1px; }}
    .percentage {{ color: #FFFFFF !important; font-size: 14px; margin-top: 4px; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="editorial-collage">
    {html_collage_images}
    <div class="title-container">
        <div class="main-title">Aura</div>
        <div class="editorial-subtitle">the fashion oracle</div>
    </div>
</div>
<p style='text-align:center; letter-spacing: 1px; filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.9)); margin-bottom: 30px; color: #FFFFFF;'>
    upload a runway or street style photo. aura segments the silhouette and reads its chromatic dna.
</p>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# VISION  (pure functions — no Streamlit rendering inside)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_segmentation_model():
    # NOTE: yolov8n-seg is trained on COCO (80 generic classes). It reliably finds
    # "person", "handbag", "tie" — it has no garment classes. To get real per-garment
    # detection, swap in a checkpoint fine-tuned on DeepFashion2 or Fashionpedia.
    return YOLO("yolov8n-seg.pt")


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def dominant_colors(pixels, k=NUM_COLORS):
    """Return (hex_codes, share_percentages) ordered most- to least-dominant."""
    unique = np.unique(pixels, axis=0)
    k = min(k, len(unique))
    if k == 0:
        return [], []

    # KMeans on a few million pixels is needlessly slow; a random sample is
    # statistically indistinguishable for dominant-colour extraction.
    if len(pixels) > KMEANS_SAMPLE_SIZE:
        idx = np.random.default_rng(42).choice(len(pixels), KMEANS_SAMPLE_SIZE, replace=False)
        pixels = pixels[idx]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_, minlength=k)
    order = np.argsort(counts)[::-1]
    centers = kmeans.cluster_centers_[order]
    counts = counts[order]

    total = counts.sum()
    return [rgb_to_hex(c) for c in centers], [(c / total) * 100 for c in counts]


def analyse_image(image_np):
    """Segment the image and extract a colour palette per detected region.

    Returns a list of dicts. Contains no Streamlit calls so it can be cached
    and so the result is a plain, serialisable value.
    """
    model = load_segmentation_model()
    # retina_masks=True returns masks at the original image resolution, which
    # avoids the misalignment caused by resizing the letterboxed 640x640 masks.
    results = model(image_np, retina_masks=True, verbose=False)
    result = results[0]

    if result.masks is None:
        return []

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.data.cpu().numpy()
    class_names = result.names

    h, w = image_np.shape[:2]
    items = []

    for i, mask in enumerate(masks):
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        pixels = image_np[mask > 0.5]
        if len(pixels) < MIN_PIXELS_PER_ITEM:
            continue

        hex_codes, shares = dominant_colors(pixels)
        if not hex_codes:
            continue

        items.append({
            "item": class_names[int(boxes[i][5])],
            "confidence": float(boxes[i][4]),
            "colors": hex_codes,
            "shares": shares,
            "area_px": int(len(pixels)),
        })

    items.sort(key=lambda d: d["area_px"], reverse=True)
    return items


def render_palette(items):
    """Draw the analysis. Reads state only — safe to call on every rerun."""
    if not items:
        st.warning("no distinct regions detected. try a clearer, full-body photo.")
        return

    for item in items:
        st.markdown(
            f"<h4 style='text-align:center; margin-top:20px; color:#FFFFFF;'>"
            f"detected: {item['item'].lower()} ({item['confidence']:.0%})</h4>",
            unsafe_allow_html=True,
        )
        cards = '<div class="palette-container">'
        for hex_code, share in zip(item["colors"], item["shares"]):
            cards += f"""<div class="color-card">
            <div class="color-box" style="background-color: {hex_code};"></div>
            <div class="color-info">
            <div class="hex-code">{hex_code.upper()}</div>
            <div class="percentage">{share:.1f}%</div>
            </div>
            </div>"""
        cards += "</div>"
        st.markdown(cards, unsafe_allow_html=True)


def build_context(items):
    if not items:
        return ""
    lines = ["outfit detected via computer vision:"]
    for item in items:
        lines.append(f"- region: {item['item']}, dominant colors (hex): {', '.join(item['colors'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "you are aura, the supreme oracle of fashion, costume, and textile history. "
    "you know every historical archive (from schiaparelli to mcqueen, alaïa to margiela), "
    "every fiber and material (duchess silk, organza, vinyl, bouclé tweed, gabardine), "
    "and you reply always and only in lowercase, with a sophisticated, dark, erudite, "
    "and sharp tone. if the user provides data about an analyzed outfit, take it into account."
)


def call_openai(messages):
    """Single entry point to the API. Raises on failure; callers decide what to show."""
    import openai
    client = openai.OpenAI(api_key=get_api_key())
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content.lower()


def opening_comment(context):
    prompt = (
        "you are aura, supreme fashion oracle. the user just uploaded an outfit. "
        f"here is the computer vision analysis of the regions and colors: {context}. "
        "write a short, sharp, dark, sophisticated opening comment on this exact chromatic "
        "combination and silhouette, ending with a question about how it feels. "
        "reply strictly in lowercase."
    )
    return call_openai([{"role": "system", "content": prompt}])


def oracle_response(context, history):
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\ncurrent outfit context: {context}"}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    return call_openai(messages)


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

st.session_state.setdefault("messages", [
    {"role": "assistant", "content": "welcome. upload your look and let's decode its aura."}
])
st.session_state.setdefault("outfit_context", "")
st.session_state.setdefault("detected_items", [])
st.session_state.setdefault("analysed_file", None)

# ---------------------------------------------------------------------------
# MAIN FLOW
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader("upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="original photography", use_container_width=True)

    st.markdown('<div class="zebra-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:center; text-transform:lowercase; letter-spacing: 4px; "
        "filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.9)); color: #FFFFFF;'>"
        "chromatic dna</h3>",
        unsafe_allow_html=True,
    )

    # This is the key fix: identify the file, and only run the expensive vision
    # pipeline when it is a file we have not already processed. Streamlit reruns
    # the whole script on every widget interaction, so without this guard every
    # chat message would re-run YOLO and KMeans from scratch.
    file_id = f"{uploaded_file.name}:{uploaded_file.size}"

    if st.session_state.analysed_file != file_id:
        with st.spinner("⏳ aura is reading the silhouette..."):
            image_np = np.array(original_image)
            st.session_state.detected_items = analyse_image(image_np)
            st.session_state.outfit_context = build_context(st.session_state.detected_items)
            st.session_state.analysed_file = file_id
            # A new image resets the conversation so the oracle is never talking
            # about a previous outfit.
            st.session_state.messages = [
                {"role": "assistant", "content": "welcome. upload your look and let's decode its aura."}
            ]

            if st.session_state.outfit_context and get_api_key():
                try:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": opening_comment(st.session_state.outfit_context)}
                    )
                except Exception:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "i have decoded the chromatic dna of your look. "
                                   "the interplay of these tones creates a striking aura.",
                    })

    render_palette(st.session_state.detected_items)

st.markdown('<div class="zebra-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="chopin-silver-title">Consult the Oracle</div>', unsafe_allow_html=True)

if not get_api_key():
    st.info(
        "no openai key found — the oracle is in offline mode. add OPENAI_API_KEY to "
        "`.streamlit/secrets.toml` or your environment to enable it."
    )

for message in st.session_state.messages:
    avatar = ai_avatar if message["role"] == "assistant" else user_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("ask aura about styling, matching, or fashion history..."):
    prompt = prompt.lower()
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ai_avatar):
        if not get_api_key():
            reply = (
                f"[offline oracle mode]: i received '{prompt}', but the archives are "
                "sealed without an api key."
            )
        else:
            with st.spinner("⏳ aura is channeling the archives..."):
                try:
                    reply = oracle_response(
                        st.session_state.outfit_context, st.session_state.messages
                    )
                except Exception as exc:
                    reply = f"the oracle senses an interference with the network: {exc}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
