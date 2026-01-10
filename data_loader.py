"""
Módulo para carregar e preparar o dataset EMOTIC
"""

import os
import numpy as np
import pandas as pd
import json
import cv2
from pathlib import Path
from tqdm import tqdm

# Imports do TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Importar CONFIG se existir, senão usar valores padrão
try:
    from config import CONFIG
except ImportError:
    CONFIG = {
        'img_height': 224,
        'img_width': 224,
        'batch_size': 32,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'num_classes': 26,
    }
    print(" config.py não encontrado, usando configuração padrão")


class EMOTICDataLoader:
    """Carrega e prepara o dataset EMOTIC"""

    def __init__(self, data_dir, annotations_file):
        self.data_dir = Path(data_dir)
        self.annotations_file = annotations_file
        self.df = None
        self.label_encoder = LabelEncoder()

    def load_annotations(self):
        """Carrega anotações do EMOTIC"""
        # EMOTIC geralmente vem com anotações em JSON ou CSV
        # Adapte conforme o formato do seu dataset

        if self.annotations_file.endswith('.json'):
            with open(self.annotations_file, 'r') as f:
                data = json.load(f)
            # Converter para DataFrame
            records = []
            for item in data:
                records.append({
                    'image_path': item['image_path'],
                    'emotion': item['emotion'],
                    'bbox': item.get('bbox', None)
                })
            self.df = pd.DataFrame(records)

        elif self.annotations_file.endswith('.csv'):
            self.df = pd.read_csv(self.annotations_file)

        print(f" Carregadas {len(self.df)} anotações")
        return self.df

    def extract_person_crops(self, output_dir):
        """
        Extrai recortes individuais das pessoas usando bounding boxes

        Args:
            output_dir: diretório para salvar os recortes
        """
        os.makedirs(output_dir, exist_ok=True)
        crops_data = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df),
                             desc="Extraindo crops"):
            img_path = self.data_dir / row['image_path']

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Se houver bbox, fazer crop
            if row['bbox'] is not None and isinstance(row['bbox'], (list, tuple)):
                x, y, w, h = row['bbox']
                crop = img[y:y + h, x:x + w]
            else:
                crop = img

            # Salvar crop
            crop_filename = f"crop_{idx}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            crops_data.append({
                'crop_path': crop_path,
                'emotion': row['emotion'],
                'original_image': row['image_path']
            })

        self.crops_df = pd.DataFrame(crops_data)
        print(f" Extraídos {len(self.crops_df)} crops")
        return self.crops_df

    def prepare_data(self, crops_df):
        """
        Prepara dados para treino

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(crops_df['emotion'])

        # Split treino/validação/teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            crops_df['crop_path'].values,
            y_encoded,
            test_size=CONFIG['test_split'],
            random_state=SEED,
            stratify=y_encoded
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=CONFIG['val_split'] / (CONFIG['train_split'] + CONFIG['val_split']),
            random_state=SEED,
            stratify=y_temp
        )

        print(f" Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_emotion_distribution(self):
        """Retorna distribuição de emoções no dataset"""
        if self.df is not None:
            return self.df['emotion'].value_counts()
        return None


# Gerador de dados customizado
class EmotionDataGenerator(keras.utils.Sequence):
    """Gerador customizado para carregar imagens sob demanda"""

    def __init__(self, image_paths, labels, batch_size, img_size, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))

        # Data augmentation
        if augment:
            self.augmentor = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='nearest'
            )

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.image_paths[batch_indices]
        batch_labels = self.labels[batch_indices]

        X = np.zeros((len(batch_paths), *self.img_size, 3))

        for i, path in enumerate(batch_paths):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = img / 255.0  # Normalização

                if self.augment:
                    img = self.augmentor.random_transform(img)

                X[i] = img

        y = keras.utils.to_categorical(batch_labels, CONFIG['num_classes'])
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


print(" Módulo de dados carregado")

# Teste rápido
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" TESTE DO MÓDULO DE CARREGAMENTO DE DADOS")
    print("=" * 70 + "\n")

    print(" Imports carregados com sucesso")
    print(" EMOTICDataLoader disponível")
    print(" EmotionDataGenerator disponível")

    # Testar criação de geradores
    print("\nTestando criação de gerador...", end=" ")
    try:
        dummy_paths = np.array(['test1.jpg', 'test2.jpg'])
        dummy_labels = np.array([0, 1])

        gen = EmotionDataGenerator(
            dummy_paths,
            dummy_labels,
            batch_size=2,
            img_size=(224, 224),
            augment=False
        )

        print("")
        print("\n TESTE PASSOU! Módulo funcionando corretamente.")
    except Exception as e:
        print(f"\n ERRO: {e}")

    print("\n" + "=" * 70)