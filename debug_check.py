import os
import sys

print("--- TANI BAŞLIYOR ---")
print(f"Şu anki konum (CWD): {os.getcwd()}")

# 1. Klasörler gerçekten orada mı?
if os.path.exists('models'):
    print("✔ 'models' klasörü mevcut.")
    print(f"   İçerik: {os.listdir('models')}")
else:
    print("❌ 'models' klasörü BURADA YOK!")

if os.path.exists('utils.py'): # Demo.py bunu kullanıyor
    print("✔ 'utils.py' mevcut.")
else:
    print("❌ 'utils.py' BURADA YOK! (Bu dosya eksikse import hatası verir)")

# 2. Yolu ekle
sys.path.append(os.getcwd())

# 3. Korumasız Import Denemesi (Hata varsa patlasın)
print("\n--- IMPORT DENEMESİ ---")
print("from models.basic_model import CDEvaluator çalıştırılıyor...")

from models.basic_model import CDEvaluator

print("✔ BAŞARILI! Import hatasız gerçekleşti.")
print("--- TANI BİTTİ ---")