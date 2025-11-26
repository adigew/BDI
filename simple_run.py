import sys
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. PATH AYARLARI ---
# Python'un 'models' klasörünü görebilmesi için
sys.path.append(os.getcwd())

# --- 2. IMPORT KONTROLÜ ---
try:
    from models.networks import BASE_Transformer, init_net
except ImportError:
    print("HATA: 'models.networks' modülü bulunamadı.")
    print("Lütfen terminali 'BIT_CD' klasöründe açtığınızdan emin olun.")
    sys.exit(1)

# --- 3. AYARLAR ---
CHECKPOINT_PATH = './checkpoints/BIT_LEVIR/best_ckpt.pt' 
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    print(f"Model yükleniyor (Cihaz: {DEVICE})...")
    
    # --- DÜZELTİLMİŞ HİBRİT MİMARİ ---
    # Analiz Sonucu:
    # 1. Encoder (transformer) BÜYÜK olmalı -> dim_head=64
    # 2. Decoder (transformer_decoder) KÜÇÜK olmalı -> decoder_dim_head=8
    model = BASE_Transformer(
        input_nc=3, 
        output_nc=2, 
        token_len=4, 
        resnet_stages_num=4,
        with_pos='learned', 
        enc_depth=1, 
        dec_depth=8,
        
        # İŞTE KİLİT NOKTA BURASI:
        dim_head=64,        # Encoder için varsayılan değer (64)
        decoder_dim_head=8  # Decoder için küçültülmüş değer (8)
    )
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint okunuyor: {CHECKPOINT_PATH}")
        # PyTorch 2.6+ Güvenlik kilidi açık
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        
        # Anahtar seçimi
        if 'model_G_state_dict' in checkpoint:
            state_dict = checkpoint['model_G_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        try:
            # strict=True yaparak her şeyin %100 uyuştuğundan emin olabiliriz artık
            model.load_state_dict(state_dict, strict=True)
            print("✔ MÜKEMMEL: Model ağırlıkları hatasız yüklendi!")
        except Exception as e:
            print(f"❌ Ağırlık yükleme hatası: {e}")
            print("Model rastgele ağırlıklarla başlatılıyor (Sonuç kötü olabilir).")
    else:
        print(f"UYARI: Checkpoint dosyası bulunamadı! Rastgele ağırlıklar kullanılıyor.")
        init_net(model, init_type='normal', init_gain=0.02, gpu_ids=[])

    model.to(DEVICE)
    model.eval()
    return model

def preprocess(img_path):
    """Görüntüyü 256x256 yapıp Tensor'a çevirir"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(img_path).convert('RGB')
    # Orijinal görüntüyü de görselleştirme için döndürüyoruz
    return transform(img).unsqueeze(0).to(DEVICE), img.resize((IMG_SIZE, IMG_SIZE))

def run_inference(pre_path, post_path):
    model = load_model()
    
    # Hazırlık
    pre_tensor, pre_vis = preprocess(pre_path)
    post_tensor, post_vis = preprocess(post_path)
    
    print("Tahmin yapılıyor...")
    with torch.no_grad():
        output = model(pre_tensor, post_tensor)
        
        # --- ÇIKTI İŞLEME DÜZELTMESİ ---
        # Output Shape: (Batch, Channel, Height, Width) -> (1, 2, 256, 256)
        probs = torch.sigmoid(output).squeeze().cpu().numpy() # -> (2, 256, 256)
        
        # Channel 0: Değişim Yok Olasılığı
        # Channel 1: Değişim VAR Olasılığı -> Biz bunu alıyoruz
        change_prob_map = probs[1, :, :] 
    
    # Binary Maske (0.5 Eşik Değeri)
    binary_mask = (change_prob_map > 0.5).astype(np.uint8) * 255
    
    # --- GÖRSELLEŞTİRME ---
    plt.figure(figsize=(15, 6))
    
    # 1. Pre
    plt.subplot(1, 3, 1)
    plt.title("Pre (Önce)")
    plt.imshow(pre_vis)
    plt.axis('off')
    
    # 2. Post
    plt.subplot(1, 3, 2)
    plt.title("Post (Sonra)")
    plt.imshow(post_vis)
    plt.axis('off')
    
    # 3. AI Maske
    plt.subplot(1, 3, 3)
    plt.title("AI Change Mask")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    print("Sonuç penceresi açılıyor...")
    plt.show()

if __name__ == '__main__':
    # --- DOSYA YOLLARI ---
    # Lütfen kendi test dosya yollarınızı buraya yazın
    my_pre_img = 'samples/A/test_113_0256.png'  #test_113_0256
    my_post_img = 'samples/B/test_113_0256.png'
    
    if os.path.exists(my_pre_img) and os.path.exists(my_post_img):
        run_inference(my_pre_img, my_post_img)
    else:
        print("HATA: Görüntü dosyaları bulunamadı.")
        print(f"Aranan yollar:\n{my_pre_img}\n{my_post_img}")