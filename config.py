import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0

# Métricas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Deteção facial
import mediapipe as mp

# Configurações
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Parâmetros do projeto
CONFIG = {
    'img_height': 224,
    'img_width': 224,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'num_classes': 26,  # EMOTIC tem 26 categorias emocionais
}

print(" Configuração inicializada")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponível: {tf.config.list_physical_devices('GPU')}")