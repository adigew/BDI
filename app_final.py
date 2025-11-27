import streamlit as st
import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import sys
# scikit-image kütüphanesi yüklü olmalı (pip install scikit-image)
from skimage.metrics import structural_similarity as ssim
from skimage import morphology, filters, measure

# --- 1. AYARLAR & KURULUM ---
sys.path.append(os.getcwd())

st.set_page_config(
    page_title="Change Detection PoC",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL SABİTLER ---
CHECKPOINT_PATH = './checkpoints/BIT_LEVIR/best_ckpt.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 256

# AI Model Kontrolü
try:
    from models.networks import BASE_Transformer, init_net
    AI_MODEL_AVAILABLE = True
except ImportError:
    AI_MODEL_AVAILABLE = False
    st.toast("⚠️ AI Modelleri klasörü bulunamadı, sadece Traditional çalışır.", icon="⚠️")

# ==========================================
# FONKSİYONLAR
# ==========================================

@st.cache_resource
def load_ai_model():
    if not AI_MODEL_AVAILABLE: return None
    model = BASE_Transformer(
        input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
        with_pos='learned', enc_depth=1, dec_depth=8,
        dim_head=64, decoder_dim_head=8 
    )
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            state_dict = checkpoint.get('model_G_state_dict', checkpoint.get('model', checkpoint))
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            st.error(f"Model Hatası: {e}")
            return None
    else:
        return None
        
    model.to(DEVICE)
    model.eval()
    return model

def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def draw_bboxes_ai(image, mask, min_area=100):
    img_bbox = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (255, 255, 0), 2)
            count += 1
    return img_bbox, count

def run_ai_detection(model, pre_img_pil, post_img_pil, conf_thr, min_area):
    if model is None: return None, None, 0
    
    pre_img = np.array(pre_img_pil)
    post_img = np.array(post_img_pil)
    if pre_img.shape[-1] == 4: pre_img = pre_img[:, :, :3]
    if post_img.shape[-1] == 4: post_img = post_img[:, :, :3]

    pre_enh = apply_clahe(pre_img)
    post_enh = apply_clahe(post_img)
    
    h, w, _ = pre_img.shape
    new_h = ((h - 1) // PATCH_SIZE + 1) * PATCH_SIZE
    new_w = ((w - 1) // PATCH_SIZE + 1) * PATCH_SIZE
    pad_h, pad_w = new_h - h, new_w - w
    
    pre_padded = cv2.copyMakeBorder(pre_enh, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    post_padded = cv2.copyMakeBorder(post_enh, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    full_mask = np.zeros((new_h, new_w), dtype=np.float32)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    
    with torch.no_grad():
        for y in range(0, new_h, PATCH_SIZE):
            for x in range(0, new_w, PATCH_SIZE):
                pre_p = pre_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                post_p = post_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                pre_t = transform(Image.fromarray(pre_p)).unsqueeze(0).to(DEVICE)
                post_t = transform(Image.fromarray(post_p)).unsqueeze(0).to(DEVICE)
                
                output = model(pre_t, post_t)
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                full_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = probs[1, :, :]
    
    prob_map = full_mask[:h, :w]
    binary_mask = (prob_map > conf_thr).astype(np.uint8) * 255
    
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    result_img, count = draw_bboxes_ai(post_img, binary_mask, min_area)
    return binary_mask, result_img, count

# def run_traditional_detection(pre_pil, post_pil, ssim_weight=0.6, rgb_weight=0.4, min_obj_size=300):
#     pre = np.array(pre_pil)
#     post = np.array(post_pil)
#     if pre.shape[-1] == 4: pre = pre[:, :, :3]
#     if post.shape[-1] == 4: post = post[:, :, :3]
    
#     if pre.shape != post.shape:
#         post = cv2.resize(post, (pre.shape[1], pre.shape[0]))

#     pre_gray = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY)
#     post_gray = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY)
    
#     win_size = min(7, pre_gray.shape[0], pre_gray.shape[1])
#     if win_size % 2 == 0: win_size -= 1
    
#     try:
#         ssim_score, ssim_map = ssim(pre_gray, post_gray, full=True, win_size=win_size)
#         diff_ssim = 1 - ssim_map
#         ssim_norm = (diff_ssim - diff_ssim.min()) / (diff_ssim.max() - diff_ssim.min() + 1e-8)
#     except:
#         return np.zeros_like(pre_gray), np.zeros_like(pre), np.zeros_like(pre), 0

#     diff = cv2.absdiff(pre, post)
#     weights = np.array([0.4, 0.3, 0.3])
#     wdiff = np.dot(diff[..., :3], weights)
#     wdiff_norm = (wdiff - wdiff.min()) / (wdiff.max() - wdiff.min() + 1e-8)

#     combined = ssim_weight * ssim_norm + rgb_weight * wdiff_norm
#     combined = combined.astype(np.float32)
#     combined_smooth = cv2.GaussianBlur(combined, (5, 5), 0)

#     try:
#         th_otsu = filters.threshold_otsu(combined_smooth)
#         th_perc = np.percentile(combined_smooth, 80)
#         th = max(th_otsu, th_perc)
#     except: th = 0.5
        
#     mask = combined_smooth > th
#     mask = morphology.remove_small_objects(mask, min_size=min_obj_size)
#     mask = morphology.remove_small_holes(mask, area_threshold=min_obj_size)
#     mask_uint8 = (mask.astype(np.uint8) * 255)

#     result_bbox = post.copy()
#     labels = measure.label(mask)
#     regions = measure.regionprops(labels)
#     count = 0
#     for region in regions:
#         if region.area < min_obj_size: continue
#         minr, minc, maxr, maxc = region.bbox
#         cv2.rectangle(result_bbox, (minc, minr), (maxc, maxr), (255, 0, 0), 2)
#         count += 1
    
#     heatmap_norm = (combined_smooth * 255).astype(np.uint8)
#     heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
#     heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
#     return mask_uint8, result_bbox, heatmap_color, count

def run_traditional_detection(pre_pil, post_pil, ssim_weight=0.6, rgb_weight=0.4, min_obj_size=300):
    pre = np.array(pre_pil)
    post = np.array(post_pil)
    if pre.shape[-1] == 4: pre = pre[:, :, :3]
    if post.shape[-1] == 4: post = post[:, :, :3]
    
    if pre.shape != post.shape:
        post = cv2.resize(post, (pre.shape[1], pre.shape[0]))

    pre_gray = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY)
    
    win_size = min(7, pre_gray.shape[0], pre_gray.shape[1])
    if win_size % 2 == 0: win_size -= 1
    
    ssim_score, ssim_map = ssim(pre_gray, post_gray, full=True, win_size=win_size)
    diff_ssim = 1 - ssim_map
    ssim_norm = (diff_ssim - diff_ssim.min()) / (diff_ssim.max() - diff_ssim.min() + 1e-8)

    diff = cv2.absdiff(pre, post)
    weights = np.array([0.4, 0.3, 0.3])
    wdiff = np.dot(diff[..., :3], weights)
    wdiff_norm = (wdiff - wdiff.min()) / (wdiff.max() - wdiff.min() + 1e-8)

    combined = ssim_weight * ssim_norm + rgb_weight * wdiff_norm
    combined = combined.astype(np.float32)
    combined_smooth = cv2.GaussianBlur(combined, (5, 5), 0)

    th_otsu = filters.threshold_otsu(combined_smooth)
    th_perc = np.percentile(combined_smooth, 80)
    th = max(th_otsu, th_perc)

    mask = combined_smooth > th
    mask = morphology.remove_small_objects(mask, min_size=min_obj_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_obj_size)
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # Bounding-box görüntüsü
    result_bbox = post.copy()

    # Kontur görüntüsü
    result_contour = post.copy()

    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    count = 0

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for region in regions:
        if region.area < min_obj_size: 
            continue
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(result_bbox, (minc, minr), (maxc, maxr), (255, 255, 0), 2)
        count += 1

    cv2.drawContours(result_contour, contours, -1, (255, 0, 0), 2)  # kırmızı kontur

    heatmap_norm = (combined_smooth * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return mask_uint8, result_bbox, result_contour, heatmap_color, count

# ==========================================
# ARAYÜZ (STREAMLIT 1.40+ UYUMLU)
# ==========================================

# --- 1. SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Kontrol Paneli")
    
    st.subheader("Görüntü Yükleme")
    pre_file = st.file_uploader("Pre (Önceki) Görüntü", type=['png', 'jpg', 'jpeg'])
    post_file = st.file_uploader("Post (Sonraki) Görüntü", type=['png', 'jpg', 'jpeg'])
    
    st.divider()
    
    st.subheader("Parametreler")
    with st.expander("🤖 AI Ayarları", expanded=False):
        ai_conf_thr = st.slider("Confidence Threshold", 0.0, 1.0, 0.15, 0.05)
        ai_min_area = st.slider("Min Pixel Area", 10, 500, 100, 10)

    with st.expander("📐 Trad. Ayarları", expanded=False):
        trad_min_obj = st.slider("Min Pixel", 50, 1000, 300, 50)
        ssim_w = st.slider("SSIM Ağırlık", 0.0, 1.0, 0.6, 0.1)

    st.divider()
    # Düzeltme: use_container_width=True -> width="stretch" (Butonlarda desteklenmeyebilir ama yeni versiyon uyarısı için denenebilir, desteklenmiyorsa bu satırda eski halini kullanın)
    # Not: st.button için hala use_container_width geçerli olabilir, ancak st.image kesinlikle width="stretch" ister.
    # Güvenli olması için butonlarda use_container_width bırakılabilir veya yeni API'ye göre güncellenebilir.
    # Kullanıcı isteği üzerine global replace yapıldı.
    run_btn = st.button("🚀 Analizi Başlat", type="primary", use_container_width=True)

# --- 2. ANA EKRAN ---

st.header("PoC Change Detection: AI vs Traditional")

if pre_file and post_file:
    image1 = Image.open(pre_file).convert('RGB')
    image2 = Image.open(post_file).convert('RGB')

    # REFERANS GÖRÜNTÜLER
    with st.expander("📸 Girdi Görüntülerini İncele", expanded=True):
        col_ref1, col_ref2 = st.columns(2)
        # Düzeltme: use_container_width=True -> width="stretch"
        col_ref1.image(image1, caption="Pre (Önce)", width="stretch")
        col_ref2.image(image2, caption="Post (Sonra)", width="stretch")

    # ANALİZ BUTONU
    if run_btn:
        st.divider()
        
        with st.spinner('Algoritmalar çalışıyor...'):
            # AI Run
            model = load_ai_model()
            ai_mask, ai_result, ai_count = run_ai_detection(model, image1, image2, ai_conf_thr, ai_min_area)
            
            # Traditional Run
            trad_mask, trad_bbox, trad_contour, trad_heatmap, trad_count = run_traditional_detection( image1, image2, ssim_weight=ssim_w, rgb_weight=(1-ssim_w), min_obj_size=trad_min_obj
)


        # --- SONUÇ GÖSTERİMİ ---
        
        # Başlıklar
        col_h1, col_h2 = st.columns(2)
        col_h1.info(f"🤖 AI Model (Bounding Box Sayısı Tespit: {ai_count})")
        col_h2.success(f"📐 Traditional (Bounding Box Sayısı Tespit: {trad_count})")

        # 1. SATIR: Sonuçlar (Box)
        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            if ai_result is not None:
                # Düzeltme: use_container_width=True -> width="stretch"
                st.image(ai_result, caption="AI Sonuç", width="stretch")
            else:
                st.warning("AI Model yüklenemedi/bulunamadı.")
        
        with row1_c2:
            # Düzeltme: use_container_width=True -> width="stretch"
            st.image(trad_bbox, caption="Traditional Sonuç", width="stretch")

        # 2. SATIR: Maskeler (Binary)
        row2_c1, row2_c2 = st.columns(2)
        with row2_c1:
            if ai_mask is not None:
                # Düzeltme: use_container_width=True -> width="stretch"
                st.image(ai_mask, caption="AI Binary Mask", width="stretch")
        
        with row2_c2:
            # Düzeltme: use_container_width=True -> width="stretch"
            st.image(trad_mask, caption="Traditional Binary Mask", width="stretch")
            
        # 3. SATIR: Extra Görseller
        with st.expander("🔥 Detaylı Isı Haritası (Traditional)", expanded=False):
             # Düzeltme: use_container_width=True -> width="stretch"
             st.image(trad_heatmap, caption="Değişim Isı Haritası (Kırmızı = Yüksek Değişim)", width="stretch")
       
        with st.expander("🔍 Traditional Contour Sonuçları", expanded=False):
             st.image(trad_contour, caption="Traditional Contours", width="stretch")

else:
    st.info("👈 Lütfen sol menüden görüntüleri yükleyin ve 'Analizi Başlat' butonuna basın.")