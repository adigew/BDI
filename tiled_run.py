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
    print("HATA: 'models' klasörü bulunamadı. Lütfen terminali 'BIT_CD' klasöründe açın.")
    sys.exit(1)

# --- AYARLAR ---
CHECKPOINT_PATH = './checkpoints/BIT_LEVIR/best_ckpt.pt' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 256  # Modelin eğitim boyutu (Sabit kalmalı)

def load_model():
    print(f"Model yükleniyor (Cihaz: {DEVICE})...")
    # Hibrit Mimari Ayarı (Doğruladığımız ayarlar)
    model = BASE_Transformer(
        input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
        with_pos='learned', enc_depth=1, dec_depth=8,
        dim_head=64,        # Encoder: 64
        decoder_dim_head=8  # Decoder: 8
    )
    
    if os.path.exists(CHECKPOINT_PATH):
        # PyTorch 2.6+ Güvenlik kilidi açık
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        # Anahtar bulma mantığı
        state_dict = checkpoint.get('model_G_state_dict', checkpoint.get('model', checkpoint))
        
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✔ Model ağırlıkları başarıyla yüklendi.")
        except Exception as e:
            print(f"❌ Ağırlık hatası: {e}")
            sys.exit(1)
    else:
        print("HATA: Checkpoint dosyası yok!")
        sys.exit(1)

    model.to(DEVICE)
    model.eval()
    return model

def inference_tiled(model, pre_path, post_path):
    """
    Görüntüyü 256x256 parçalara bölerek analiz eder (Resize yok).
    """
    # 1. Görüntüleri Oku
    pre_img = cv2.imread(pre_path)
    post_img = cv2.imread(post_path)
    
    # BGR -> RGB Dönüşümü
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
    
    h, w, c = pre_img.shape
    print(f"Orijinal Boyut: {w}x{h}")
    
    # 2. Padding (256'nın katlarına tamamla)
    # Örn: 384 -> 512'ye tamamlanacak (128 piksel siyah eklenecek)
    stride = PATCH_SIZE # Kaydırma miktarı
    
    # Yeni boyutları hesapla (256'nın katı olacak şekilde yukarı yuvarla)
    new_h = ((h - 1) // PATCH_SIZE + 1) * PATCH_SIZE
    new_w = ((w - 1) // PATCH_SIZE + 1) * PATCH_SIZE
    
    # Padding ekle (Sağ ve Alt tarafa)
    pad_h = new_h - h
    pad_w = new_w - w
    
    pre_padded = cv2.copyMakeBorder(pre_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    post_padded = cv2.copyMakeBorder(post_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    full_mask = np.zeros((new_h, new_w), dtype=np.float32)
    
    # Dönüştürücü (Sadece Tensor, Resize YOK)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print(f"İşlenen Boyut (Padded): {new_w}x{new_h} -> Tiling Başlıyor...")
    
    # 3. Döngü (Tiling)
    with torch.no_grad():
        for y in range(0, new_h, stride):
            for x in range(0, new_w, stride):
                # Parçayı kes
                pre_patch = pre_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                post_patch = post_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                # Modele ver
                pre_t = transform(Image.fromarray(pre_patch)).unsqueeze(0).to(DEVICE)
                post_t = transform(Image.fromarray(post_patch)).unsqueeze(0).to(DEVICE)
                
                output = model(pre_t, post_t)
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Değişim kanalını al
                patch_mask = probs[1, :, :]
                
                # Büyük maskeye yapıştır
                full_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = patch_mask

    # 4. Padding'i kesip at (Orijinal boyuta dön)
    final_mask = full_mask[:h, :w]
    
    return pre_img, post_img, final_mask

if __name__ == '__main__':
    # --- DOSYA YOLLARI ---
    # Senin paylaştığın örnek dosya isimleri
    my_pre_img = 'samples/A/test_113_0256.png'  
    my_post_img = 'samples/B/test_113_0256.png'
    
    if os.path.exists(my_pre_img):
        model = load_model()
        pre, post, prob_map = inference_tiled(model, my_pre_img, my_post_img)
        
        # --- Son İşlemler (Post-Processing) ---
        # 1. Eşik Değeri (Threshold): 0.4 (Biraz hassas olsun)
        binary_mask = (prob_map > 0.4).astype(np.uint8) * 255 
        
        # 2. Gürültü Temizleme (Morphology)
        kernel = np.ones((3,3), np.uint8)
        # Küçük beyaz noktaları sil
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Kopuk parçaları birleştir
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # --- GÖRSELLEŞTİRME ---
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        plt.title("Pre (Önce)")
        plt.imshow(pre)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Post (Sonra)")
        plt.imshow(post)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("AI Maske (Tiled)")
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        print("Sonuç penceresi açılıyor...")
        plt.show()
    else:
        print("HATA: Dosya bulunamadı.")