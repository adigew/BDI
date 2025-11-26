import torch
import os

# Dosya yolunu kontrol edin
ckpt_path = './checkpoints/BIT_LEVIR/best_ckpt.pt'

print(f"--- CHECKPOINT ANALİZİ: {ckpt_path} ---")

if not os.path.exists(ckpt_path):
    print("❌ HATA: Dosya bulunamadı!")
else:
    try:
        # CPU'ya map ederek yüklüyoruz (Hata riskini azaltır)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        
        print("✔ Dosya başarıyla yüklendi.")
        print(f"Veri Tipi: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print("\n🔑 Anahtarlar (Keys):")
            print(list(checkpoint.keys()))
            
            # İçerik kontrolü
            if 'net_G' in checkpoint:
                print("\n✅ 'net_G' anahtarı bulundu (Model ağırlıkları burada).")
            elif 'model' in checkpoint:
                print("\n✅ 'model' anahtarı bulundu.")
            elif 'state_dict' in checkpoint:
                print("\n✅ 'state_dict' anahtarı bulundu.")
            else:
                print("\n⚠️ Model ağırlıkları için standart bir anahtar bulunamadı!")
                # İlk 3 anahtarı gösterelim ki ne olduğunu anlayalım
                first_keys = list(checkpoint.keys())[:3]
                print(f"İlk anahtarlar: {first_keys}")

        # Eğer argümanlar kaydedilmişse onları da görelim
        if 'args' in checkpoint: # Bazen eğitim parametreleri de dosyaya gömülür
            print("\n⚙️ Kayıtlı Eğitim Argümanları:")
            print(checkpoint['args'])
            
    except Exception as e:
        print(f"\n❌ Dosya okuma hatası:\n{e}")