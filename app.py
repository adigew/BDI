import streamlit as st
import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import sys

# --- 1. AYARLAR & KURULUM ---
sys.path.append(os.getcwd())

# Model modüllerini import etmeye çalış
try:
    from models.networks import BASE_Transformer, init_net
except ImportError:
    st.error("HATA: 'models' klasörü bulunamadı. Lütfen terminali projenin ana dizininde açın.")
    st.stop()

# --- SABİTLER ---
CHECKPOINT_PATH = './checkpoints/BIT_LEVIR/best_ckpt.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 256

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="AI Change Detection PoC",
    page_icon="🛰️",
    layout="wide"
)

# --- 2. MODEL YÜKLEME (CACHE) ---
@st.cache_resource
def load_model():
    model = BASE_Transformer(
        input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
        with_pos='learned', enc_depth=1, dec_depth=8,
        dim_head=64, decoder_dim_head=8 
    )
    
    if os.path.exists(CHECKPOINT_PATH):
        # weights_only=False güvenlik uyarısını aşmak için
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get('model_G_state_dict', checkpoint.get('model', checkpoint))
        model.load_state_dict(state_dict, strict=True)
    else:
        st.error(f"Checkpoint bulunamadı: {CHECKPOINT_PATH}")
        st.stop()

    model.to(DEVICE)
    model.eval()
    return model

# --- 3. YARDIMCI FONKSİYONLAR ---
def apply_clahe(img_rgb):
    """Kontrast Artırma"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def draw_bboxes(image, mask, min_area=100):
    """Maskedeki beyaz alanlara sarı kutu çizer"""
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

def process_images(model, pre_img_pil, post_img_pil, conf_thr, min_area):
    # PIL -> OpenCV Formatına Çevir
    pre_img = np.array(pre_img_pil)
    post_img = np.array(post_img_pil)
    
    # RGB kontrolü
    if pre_img.shape[-1] == 4: pre_img = pre_img[:, :, :3]
    if post_img.shape[-1] == 4: post_img = post_img[:, :, :3]

    # Ön İşleme (CLAHE)
    pre_enh = apply_clahe(pre_img)
    post_enh = apply_clahe(post_img)
    
    # Tiling Hazırlığı
    h, w, _ = pre_img.shape
    new_h = ((h - 1) // PATCH_SIZE + 1) * PATCH_SIZE
    new_w = ((w - 1) // PATCH_SIZE + 1) * PATCH_SIZE
    pad_h, pad_w = new_h - h, new_w - w
    
    pre_padded = cv2.copyMakeBorder(pre_enh, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    post_padded = cv2.copyMakeBorder(post_enh, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    full_mask = np.zeros((new_h, new_w), dtype=np.float32)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    
    # İlerleme Çubuğu
    progress_bar = st.progress(0)
    total_steps = (new_h // PATCH_SIZE) * (new_w // PATCH_SIZE)
    step = 0
    
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
                
                step += 1
                progress_bar.progress(min(step / total_steps, 1.0))
    
    progress_bar.empty()

    # Orijinal boyuta dön
    prob_map = full_mask[:h, :w]
    
    # Post Processing
    binary_mask = (prob_map > conf_thr).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    result_img, count = draw_bboxes(post_img, binary_mask, min_area)
    
    return binary_mask, result_img, count

# --- 4. ARAYÜZ TASARIMI ---
st.title("🛰️ AI Change Detection PoC")
st.markdown("""
Bu demo, **Bitemporal Image Transformer (BIT)** modelini kullanarak iki zamanlı görüntü arasındaki farkları tespit eder.
""")

# Yan Menü (Ayarlar)
with st.sidebar:
    st.header("⚙️ Ayarlar")
    st.info(f"Cihaz: {DEVICE}")
    conf_thr = st.slider("Hassasiyet Eşiği (Threshold)", 0.0, 1.0, 0.15, 0.05)
    min_area = st.slider("Min Değişim Alanı (Piksel)", 10, 500, 100, 10)
    st.divider()
    st.markdown("Developed for PoC")

# Dosya Yükleme Alanı
col1, col2 = st.columns(2)
with col1:
    pre_file = st.file_uploader("1. Pre (Önceki) Görüntü", type=['png', 'jpg', 'jpeg'])
with col2:
    post_file = st.file_uploader("2. Post (Sonraki) Görüntü", type=['png', 'jpg', 'jpeg'])

# --- GÖRSELLEŞTİRME VE İŞLEM AKIŞI ---
if pre_file and post_file:
    # 1. Görüntüleri Yükle ve GÖSTER (En üstte)
    image1 = Image.open(pre_file).convert('RGB')
    image2 = Image.open(post_file).convert('RGB')
    
    st.divider()
    st.subheader("📸 Girdi Görüntüleri")
    
    # Girdileri yan yana göster
    view_c1, view_c2 = st.columns(2)
    view_c1.image(image1, caption="Pre (Önce)", width="stretch")
    view_c2.image(image2, caption="Post (Sonra)", width="stretch")
    
    st.divider()
    
    # 2. Buton ve İşlem (Ortada)
    if st.button("🚀 Analizi Başlat", type="primary"):
        with st.spinner('Yapay Zeka görüntüleri işliyor...'):
            try:
                # Modeli yükle
                model = load_model()
                
                # İşlemi yap
                mask, result, count = process_images(model, image1, image2, conf_thr, min_area)
                
                st.success(f"Analiz Tamamlandı! Toplam **{count}** adet değişim tespit edildi.")
                
                # 3. Sonuçları GÖSTER (En Altta)
                st.subheader("📊 Analiz Sonuçları")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.info("🔍 AI Maskesi (Pixel-wise)")
                    st.image(mask, caption="Yapay Zeka Tespiti (Siyah-Beyaz)", width="stretch")
                
                with res_col2:
                    st.success("🎯 Sonuç (Bounding Box)")
                    st.image(result, caption="Tespit Edilen Değişimler", width="stretch")

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
else:
    st.info("Lütfen analiz etmek için her iki görüntüyü de yükleyin.")