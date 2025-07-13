import os
import pandas as pd
import numpy as np
from glob import glob

# Proje ana dizinini al
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')

# 1. Meta veriyi yükle
metadata_path = os.path.join(dataset_dir, 'HAM10000_metadata.csv')
skin_df = pd.read_csv(metadata_path)

# 2. Görüntü yollarını oluştur ve birleştir
# Glob ile tüm görüntülerin yollarını bul
image_path_pattern = os.path.join(dataset_dir, 'HAM10000_images_part_*', '*.jpg')
image_paths = glob(image_path_pattern)

# Görüntü yollarını {image_id: path} şeklinde bir sözlüğe çevir
# Bu, arama işlemini çok daha hızlı hale getirir (O(1) karmaşıklık)
image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in image_paths}

# Meta veri DataFrame'ine 'path' sütununu ekle
skin_df['path'] = skin_df['image_id'].map(image_path_dict.get)

# 3. Etiketleri (dx) sayısal değerlere dönüştür
# Lezyon tiplerini ve kodlarını bir sözlükte tanımla
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
lesion_class_dict = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

skin_df['lesion_type'] = skin_df['dx'].map(lesion_type_dict.get)
skin_df['cell_type_idx'] = skin_df['dx'].map(lesion_class_dict.get)

# 4. Veriyi kontrol et
print("Veri setinin ilk 5 satırı:")
print(skin_df.head())
print("\nKayıp veri kontrolü:")
print(skin_df.isnull().sum())
print("\nSınıf Dağılımı:")
print(skin_df['dx'].value_counts())

# 'age' sütunundaki eksik verileri ortalama ile doldur
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)

# Tekrar kayıp veri kontrolü yapalım
print("\n'age' sütunu doldurulduktan sonra kayıp veri durumu:")
print(skin_df.isnull().sum())
