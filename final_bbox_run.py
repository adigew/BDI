import sys
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# --- PATH & IMPORT ---
sys.path.append(os.getcwd())
try:
    from models.networks import BASE_Transformer, init_net
except ImportError:
    print("HATA: 'models' klasörü bulunamadı.")
    sys.exit(1)

# --- AYARLAR ---
CHECKPOINT_PATH = './checkpoints/BIT_LEVIR/best_ckpt.pt' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 256
MIN_AREA = 100  # Bu değerden küçük değişimleri (gürültüleri) yoksay

def load_model():
    print(f"Model yükleniyor (Cihaz: {DEVICE})...")
    model = BASE_Transformer(
        input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
        with_pos='learned', enc_depth=1, dec_depth=8,
        dim_head=64, decoder_dim_head=8 
    )
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get('model_G_state_dict', checkpoint.get('model', checkpoint))
        model.load_state_dict(state_dict, strict=True)
    else:
        sys.exit(1)
    model.to(DEVICE)
    model.eval()
    return model

def apply_clahe(img_rgb):
    """Kontrast Artırma"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def draw_bboxes(image, mask):
    """Maskedeki beyaz alanlara kutu çizer"""
    # Görüntü kopyası (üzerine çizim yapacağız)
    img_bbox = image.copy()
    
    # Konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for cnt in contours:
        # ALAN FİLTRESİ: Çok küçük noktaları (gürültü) görmezden gel
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Sarı Kutu Çiz (RGB: 255, 255, 0)
            cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
            # Etiket (Opsiyonel)
            label = f"Change {int(area)}"
            # cv2.putText(img_bbox, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            count += 1
            
    return img_bbox, count

def run_final_poc(model, pre_path, post_path):
    # 1. Okuma & CLAHE
    pre_img = cv2.imread(pre_path)
    post_img = cv2.imread(post_path)
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
    
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
    
    print("AI Analizi Yapılıyor (Tiling + Enhancement)...")
    with torch.no_grad():
        for y in range(0, new_h, PATCH_SIZE):
            for x in range(0, new_w, PATCH_SIZE):
                pre_t = transform(Image.fromarray(pre_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE])).unsqueeze(0).to(DEVICE)
                post_t = transform(Image.fromarray(post_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE])).unsqueeze(0).to(DEVICE)
                
                output = model(pre_t, post_t)
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                full_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = probs[1, :, :]

    # Orijinal boyuta dön
    prob_map = full_mask[:h, :w]
    
    # --- POST PROCESSING ---
    # 1. Eşikleme (Heatmap'te gördüğümüz gibi 0.15 iyi çalışıyor)
    binary_mask = (prob_map > 0.15).astype(np.uint8) * 255
    
    # 2. Temizlik (Küçük noktaları sil, delikleri kapat)
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Bounding Box Çizimi
    result_img, box_count = draw_bboxes(post_img, binary_mask)
    
    print(f"Toplam {box_count} adet değişim tespit edildi.")
    return pre_img, post_img, result_img, binary_mask

if __name__ == '__main__':
    my_pre = 'samples/A/test_113_0256.png'  
    my_post = 'samples/B/test_113_0256.png'
    
    if os.path.exists(my_pre):
        model = load_model()
        pre, post, result, mask = run_final_poc(model, my_pre, my_post)
        
        # --- SUNUM GÖRSELİ ---
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.title("Pre Image")
        plt.imshow(pre)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Post Image")
        plt.imshow(post)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"AI Detection (BIT Transformer)")
        plt.imshow(result) # Kutulu sonuç
        plt.axis('off')
        
        plt.tight_layout()
        print("PoC Sonucu Gösteriliyor...")
        plt.show()