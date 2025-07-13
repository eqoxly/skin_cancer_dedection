import os
import pandas as pd
import numpy as np
from glob import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==================================================
# PROJE AYARLARI VE SABİTLER
# ==================================================
# Proje ana dizinini al
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

# İşlenmiş veri dosyalarının yolları
TRAIN_TENSORS_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_tensors.pt')
TEST_TENSORS_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_tensors.pt')

IMAGE_SIZE = 224
TARGET_SAMPLES = 2000 # Veri çoğaltma için hedef örnek sayısı

# Lezyon tipleri ve kodları
LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi', 'mel': 'Melanoma', 'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma', 'akiec': 'Actinic keratoses', 'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
LESION_CLASS_DICT = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}

# ==================================================
# ADIM 0: İŞLENMİŞ VERİYİ YÜKLE VEYA İŞLE
# ==================================================
# İşlenmiş veri klasörü yoksa oluştur
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

if os.path.exists(TRAIN_TENSORS_PATH) and os.path.exists(TEST_TENSORS_PATH):
    print("="*50)
    print("ADIM 0: Hazır işlenmiş veri bulundu ve yükleniyor...")
    print("="*50)
    
    train_data = torch.load(TRAIN_TENSORS_PATH)
    test_data = torch.load(TEST_TENSORS_PATH)
    
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    X_test = test_data['X_test']
    y_test = test_data['y_test']

    print("Veriler başarıyla yüklendi.")
    print(f"Eğitim seti boyutu: {X_train.shape[0]} örnek")
    print(f"Test seti boyutu: {X_test.shape[0]} örnek")
    print("\nÇoğaltma sonrası eğitim setindeki sınıf dağılımı:")
    class_counts = torch.bincount(y_train)
    for i, count in enumerate(class_counts):
        print(f"Sınıf {i}: {count.item()} örnek")

else:
    print("="*50)
    print("ADIM 0: Hazır veri bulunamadı. Veri işleme süreci başlatılıyor...")
    print("="*50)

    # ==================================================
    # ADIM 1: META VERİYİ YÜKLEME VE BİRLEŞTİRME
    # ==================================================
    print("\n" + "="*50)
    print("ADIM 1: META VERİYİ YÜKLEME VE BİRLEŞTİRME")
    print("="*50)
    metadata_path = os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')
    skin_df = pd.read_csv(metadata_path)

    image_path_pattern = os.path.join(DATASET_DIR, 'HAM10000_images_part_*', '*.jpg')
    image_paths = glob(image_path_pattern)
    image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in image_paths}

    skin_df['path'] = skin_df['image_id'].map(image_path_dict.get)
    skin_df['lesion_type'] = skin_df['dx'].map(LESION_TYPE_DICT.get)
    skin_df['cell_type_idx'] = skin_df['dx'].map(LESION_CLASS_DICT.get)
    skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
    print("Meta veri yüklendi ve işlendi.")
    print(f"Toplam {len(skin_df)} adet kayıt bulundu.")

    # ==================================================
    # ADIM 2: GÖRÜNTÜ İŞLEME VE HAZIRLIK
    # ==================================================
    print("\n" + "="*50)
    print("ADIM 2: GÖRÜNTÜ İŞLEME VE HAZIRLIK")
    print("="*50)
    tqdm.pandas(desc="Görüntüler işleniyor")
    skin_df['image_pixels'] = skin_df['path'].progress_apply(
        lambda x: np.asarray(Image.open(x).resize((IMAGE_SIZE, IMAGE_SIZE)))
    )
    skin_df['image_pixels'] = skin_df['image_pixels'].apply(lambda x: x / 255.0)
    print("\nGörüntü işleme tamamlandı.")

    # ==================================================
    # ADIM 3: EĞİTİM VE TEST SETLERİNİN OLUŞTURULMASI
    # ==================================================
    print("\n" + "="*50)
    print("ADIM 3: EĞİTİM VE TEST SETLERİNİN OLUŞTURULMASI")
    print("="*50)
    X = np.asarray(skin_df['image_pixels'].tolist())
    y = skin_df['cell_type_idx']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Veri seti eğitim ve test olarak ayrıldı.")

    # ==================================================
    # ADIM 4: VERİ ÇOĞALTMA (DATA AUGMENTATION)
    # ==================================================
    print("\n" + "="*50)
    print("ADIM 4: VERİ ÇOĞALTMA (DATA AUGMENTATION)")
    print("="*50)
    
    # NumPy'den PyTorch Tensor'larına geçiş
    X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2).float()
    X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2).float()
    y_train = torch.from_numpy(y_train.values).long()
    y_test = torch.from_numpy(y_test.values).long()
    print("Veriler PyTorch Tensor'larına dönüştürüldü.")

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])

    print(f"\nAzınlık sınıflar {TARGET_SAMPLES} örneğe tamamlanacak...")
    augmented_images = []
    augmented_labels = []

    for i in range(1, 7):
        class_indices = torch.where(y_train == i)[0]
        num_samples = len(class_indices)
        if num_samples == 0: continue
        num_to_generate = TARGET_SAMPLES - num_samples
        if num_to_generate <= 0: continue

        print(f"Sınıf {i}: {num_samples} -> {TARGET_SAMPLES} (Eklenecek: {num_to_generate})")
        class_images = X_train[class_indices]
        
        for _ in tqdm(range(num_to_generate), desc=f"Sınıf {i} çoğaltılıyor"):
            random_index = torch.randint(0, num_samples, (1,)).item()
            image_to_augment = class_images[random_index]
            augmented_image = data_transforms(image_to_augment)
            augmented_images.append(augmented_image)
            augmented_labels.append(torch.tensor(i))

    if augmented_images:
        X_train = torch.cat([X_train, torch.stack(augmented_images)], dim=0)
        y_train = torch.cat([y_train, torch.stack(augmented_labels)], dim=0)

    print("\nEğitim seti karıştırılıyor...")
    shuffle_indices = torch.randperm(X_train.shape[0])
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    print("Veri çoğaltma tamamlandı.")

    # ==================================================
    # ADIM 5: İŞLENMİŞ VERİYİ KAYDETME
    # ==================================================
    print("\n" + "="*50)
    print("ADIM 5: İŞLENMİŞ VERİYİ KAYDETME")
    print("="*50)
    
    # Eğitim ve test verilerini ayrı sözlüklerde kaydet
    torch.save({'X_train': X_train, 'y_train': y_train}, TRAIN_TENSORS_PATH)
    torch.save({'X_test': X_test, 'y_test': y_test}, TEST_TENSORS_PATH)
    
    print(f"İşlenmiş eğitim verisi şuraya kaydedildi: {TRAIN_TENSORS_PATH}")
    print(f"İşlenmiş test verisi şuraya kaydedildi: {TEST_TENSORS_PATH}")

# ==================================================
# GELECEK ADIMLAR: MODEL EĞİTİMİ
# ==================================================
print("\n" + "="*50)
print("VERİ HAZIRLIĞI TAMAMLANDI. MODEL EĞİTİMİNE HAZIR.")
print("="*50)
