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

def load_model():
    print(f"Model yükleniyor (Cihaz: {DEVICE})...")
    # Hibrit Mimari (64/8)
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
    """
    Görüntünün kontrastını yerel olarak artırır (Binayı zeminden ayırır).
    """
    # RGB -> LAB renk uzayına geç
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Sadece Işık (L) kanalına CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)) # ClipLimit'i artırdık (2->4)
    cl = clahe.apply(l)
    
    # Birleştir ve geri RGB'ye dön
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def inference_tiled_enhanced(model, pre_path, post_path):
    # 1. Oku
    pre_img = cv2.imread(pre_path)
    post_img = cv2.imread(post_path)
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
    
    # 2. ÖN İŞLEME: CLAHE (Kontrast Artırma)
    # Bu adım modelin gizli detayları görmesini sağlar
    pre_enhanced = apply_clahe(pre_img)
    post_enhanced = apply_clahe(post_img)
    
    h, w, c = pre_img.shape
    new_h = ((h - 1) // PATCH_SIZE + 1) * PATCH_SIZE
    new_w = ((w - 1) // PATCH_SIZE + 1) * PATCH_SIZE
    pad_h, pad_w = new_h - h, new_w - w
    
    # Padded görüntüler (Enhanced versiyonları kullanıyoruz!)
    pre_padded = cv2.copyMakeBorder(pre_enhanced, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    post_padded = cv2.copyMakeBorder(post_enhanced, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    full_mask = np.zeros((new_h, new_w), dtype=np.float32)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    
    print("Tiling + CLAHE Enhancement ile işleniyor...")
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

    return pre_img, post_img, full_mask[:h, :w]

if __name__ == '__main__':
    my_pre_img = 'samples/A/test_113_0256.png'  
    my_post_img = 'samples/B/test_113_0256.png'
    
    if os.path.exists(my_pre_img):
        model = load_model()
        pre, post, prob_map = inference_tiled_enhanced(model, my_pre_img, my_post_img)
        
        # --- GÖRSELLEŞTİRME ---
        plt.figure(figsize=(16, 6))
        
        # 1. Post Görüntü
        plt.subplot(1, 3, 1)
        plt.title("Post Image")
        plt.imshow(post)
        plt.axis('off')
        
        # 2. Isı Haritası (Heatmap) - Modelin ne düşündüğünü renkli görelim
        plt.subplot(1, 3, 2)
        plt.title("AI Heatmap (Kırmızı=Yüksek Olasılık)")
        plt.imshow(prob_map, cmap='jet', vmin=0, vmax=1) 
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # 3. Sonuç Maskesi (Çok düşük eşik değeri ile)
        # Eşiği 0.15'e çektik. Eğer model %15 bile şüpheleniyorsa göstersin.
        binary_mask = (prob_map > 0.15).astype(np.uint8) * 255
        
        # Temizlik
        kernel = np.ones((3,3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        plt.subplot(1, 3, 3)
        plt.title("Final Maske (Threshold: 0.15)")
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("HATA: Dosya bulunamadı.")