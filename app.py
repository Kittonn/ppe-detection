# app.py - ANTI-FLICKER VERSION
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from huggingface_hub import hf_hub_download
import hashlib
import io

# ──────────────────────────────────────────────────────────────
# CONFIG & THEME
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PPE Detection App", page_icon="🦺", layout="wide")
st.markdown("""
<style>
  [data-testid="stAppViewContainer"]{
    background: linear-gradient(180deg,#0f1117 0%,#1c1e26 100%); color:#fff;
  }
  h1,h2,h3{color:#fff;font-family:'Inter',sans-serif}
  .block-container{padding-top:1rem}
  .stSlider label,.stSelectbox label,.stFileUploader label{color:#fff}
  
  /* CRITICAL: Fix image container size to prevent layout shifts */
  [data-testid="stImage"] > div {
    height: 400px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
  }
  [data-testid="stImage"] img {
    max-height: 400px !important;
    object-fit: contain !important;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🦺 PPE Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#b0b0b0;'>Original (Left) vs Predicted (Right)</p>", unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────
def init_session_state():
    """Initialize session state variables"""
    if 'cached_results' not in st.session_state:
        st.session_state.cached_results = {}
    if 'last_file_hash' not in st.session_state:
        st.session_state.last_file_hash = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

init_session_state()

# ──────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────
def get_file_hash(uploaded_file):
    """Get hash of uploaded file for caching"""
    if uploaded_file is None:
        return None
    
    # Reset file pointer and read
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset for later use
    
    return hashlib.md5(file_bytes).hexdigest()

def resize_image_for_display(img, max_size=(800, 400)):
    """Resize image to consistent size for display"""
    w, h = img.size
    ratio = min(max_size[0]/w, max_size[1]/h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

# ──────────────────────────────────────────────────────────────
# HUGGING FACE HUB (ดาวน์โหลดโมเดลอัตโนมัติครั้งแรก)
# ──────────────────────────────────────────────────────────────
HF_REPO = "PcrPz/ppe-models"

@st.cache_resource
def ensure_from_hub(filename: str):
    """ถ้าไฟล์ยังไม่มี ให้ดึงจาก Hugging Face Hub มาเก็บโลคัล"""
    if not os.path.exists(filename):
        try:
            print(f"⬇️ Downloading {filename} from Hugging Face Hub...")
            hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=".")
            print(f"✅ Download complete: {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False
    return True

# Pre-download models
model_status = {
    'yolo': ensure_from_hub("best.pt"),
    'frcnn': ensure_from_hub("fasterrcnn_ppe_model_kfold_fold3.pth")
}

# ──────────────────────────────────────────────────────────────
# CLASSES (11 foreground classes)
# ──────────────────────────────────────────────────────────────
FRCNN_FOREGROUND_NAMES = [
    "Gloves","Goggles","Hardhat","Mask",
    "Safety Vest","Person",
    "NO-Gloves","NO-Goggles","NO-Hardhat",
    "NO-Mask","NO-Safety Vest"
]

def color_for(name:str):
    if name=="Person": return (0,102,255)      # blue
    if name.startswith("NO-"): return (255,0,0) # red
    return (57,255,20)                          # green

# ──────────────────────────────────────────────────────────────
# LOAD MODELS (cache)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo(path="best.pt"):
    if not model_status.get('yolo', False):
        st.error("YOLO model not available")
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Failed to load YOLO: {e}")
        return None

@st.cache_resource
def load_frcnn(path="fasterrcnn_ppe_model_kfold_fold3.pth"):
    if not model_status.get('frcnn', False):
        st.error("Faster R-CNN model not available")
        return None
    try:
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=12)
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load Faster R-CNN: {e}")
        return None

# ──────────────────────────────────────────────────────────────
# DRAWING UTILS
# ──────────────────────────────────────────────────────────────
def draw_boxes_pil(base_img:Image.Image, boxes_xyxy, label_ids, scores, conf_thres, names_lookup):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for (x1,y1,x2,y2), lab, sc in zip(boxes_xyxy, label_ids, scores):
        if float(sc) < float(conf_thres): 
            continue
        lab = int(lab)
        name = names_lookup(lab)
        color = color_for(name)
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        label_text = f"{name} ({float(sc):.2f})"
        tw = int(draw.textlength(label_text, font=font))
        draw.rectangle([x1, max(0,y1-20), x1+tw+6, y1], fill=color)
        draw.text((x1+3, max(0,y1-18)), label_text, fill=(0,0,0), font=font)
    return img

# ──────────────────────────────────────────────────────────────
# CACHED INFERENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_yolo_inference(image_hash: str, conf: float, _image_pil):
    """Cached YOLO inference"""
    yolo = load_yolo("best.pt")
    if yolo is None:
        return None, [], []
    
    results = yolo(_image_pil, conf=conf, imgsz=640)
    boxes = results[0].boxes

    # names lookup
    names_dict = getattr(results[0], "names", None) or getattr(yolo, "names", None)
    if isinstance(names_dict, dict) and len(names_dict) > 0:
        def yolo_lookup(cid:int): return str(names_dict.get(cid, "unknown"))
    else:
        def yolo_lookup(cid:int):
            return FRCNN_FOREGROUND_NAMES[cid] if 0 <= cid < len(FRCNN_FOREGROUND_NAMES) else "unknown"

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        pred_img = draw_boxes_pil(_image_pil, xyxy, cls_ids, scores, conf_thres=conf, names_lookup=yolo_lookup)
        labels = [yolo_lookup(i) for i in cls_ids]
        confidences = [round(float(s),2) for s in scores]
        
        return pred_img, labels, confidences
    else:
        return _image_pil, [], []

@st.cache_data(show_spinner=False)
def run_frcnn_inference(image_hash: str, conf: float, _image_pil):
    """Cached Faster R-CNN inference"""
    frcnn = load_frcnn("fasterrcnn_ppe_model_kfold_fold3.pth")
    if frcnn is None:
        return None, [], []
    
    with torch.no_grad():
        pred = frcnn([T.ToTensor()(_image_pil)])[0]

    def frcnn_lookup(cid:int):
        return FRCNN_FOREGROUND_NAMES[cid] if 0 <= cid < len(FRCNN_FOREGROUND_NAMES) else "unknown"

    if len(pred["boxes"]) == 0:
        return _image_pil, [], []
    
    xyxy_all = pred["boxes"].cpu().numpy()
    labels_all = pred["labels"].cpu().numpy().astype(int)
    scores_all = pred["scores"].cpu().numpy()

    keep = labels_all > 0            # ตัด background
    xyxy   = xyxy_all[keep]
    labels = (labels_all[keep] - 1)  # 0..10
    scores = scores_all[keep]

    pred_img = draw_boxes_pil(_image_pil, xyxy, labels, scores, conf_thres=conf, names_lookup=frcnn_lookup)
    labels_list = [frcnn_lookup(int(i)) for i in labels]
    confidences = [round(float(s),2) for s in scores]
    
    return pred_img, labels_list, confidences

# ──────────────────────────────────────────────────────────────
# SETTINGS BAR (บนสุด): Upload | Model | Threshold
# ──────────────────────────────────────────────────────────────
c0, c1, c2 = st.columns([2,1,1.2])
with c0:
    uploaded = st.file_uploader("📤 Upload Image", type=["jpg","jpeg","png"], key="file_upload")
with c1:
    model_choice = st.selectbox("🧠 Model", ["YOLOv8","Faster R-CNN"], key="model_choice")
with c2:
    conf = st.slider("🎯 Confidence Threshold", 0.10, 0.95, 0.25, 0.05, key="conf_slider")

st.divider()

# ──────────────────────────────────────────────────────────────
# MAIN INFERENCE SECTION
# ──────────────────────────────────────────────────────────────
if uploaded:
    # Get file hash for caching
    current_hash = get_file_hash(uploaded)
    
    # Load image
    pil_img = Image.open(uploaded).convert("RGB")
    display_img = resize_image_for_display(pil_img)
    
    # Update session state
    st.session_state.current_image = display_img
    st.session_state.last_file_hash = current_hash

    # Create stable layout
    left, right = st.columns(2, gap="large")
    
    # ORIGINAL IMAGE (LEFT) - Always stable
    with left:
        st.markdown("### 🖼️ Original")
        st.image(display_img, use_container_width=True)

    # PREDICTION (RIGHT)
    with right:
        st.markdown("### 🎯 Prediction")
        
        # Show spinner only during processing
        with st.spinner("🔄 Processing..."):
            try:
                if model_choice == "YOLOv8":
                    pred_img, labels, confidences = run_yolo_inference(current_hash, conf, pil_img)
                    model_name = "YOLO"
                else:
                    pred_img, labels, confidences = run_frcnn_inference(current_hash, conf, pil_img)
                    model_name = "Faster R-CNN"
                
                if pred_img is not None:
                    # Resize prediction image to same size as original
                    pred_display = resize_image_for_display(pred_img)
                    st.image(pred_display, use_container_width=True)
                    
                    # Show results below images
                    if labels:
                        st.markdown(f"##### 📄 Detected ({model_name}):")
                        for label, conf_val in zip(labels, confidences):
                            if conf_val >= conf:  # Only show above threshold
                                st.markdown(f"- **{label}** — Confidence: `{conf_val}`")
                    else:
                        st.info("No objects detected above threshold.")
                else:
                    st.error("Failed to process image")
                    st.image(display_img, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.image(display_img, use_container_width=True)

else:
    # No image uploaded - show placeholders
    left, right = st.columns(2, gap="large")
    
    with left:
        st.markdown("### 🖼️ Original")
        st.info("Upload an image to start detection")
    
    with right:
        st.markdown("### 🎯 Prediction")
        st.info("Results will appear here")
    
    st.info("👆 Upload an image above, select model and adjust threshold to start detection")
