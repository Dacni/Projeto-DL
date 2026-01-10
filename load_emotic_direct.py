"""
Carrega o dataset EMOTIC diretamente dos ficheiros .npy
SEM necessidade de conversão para JPG!
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

print("="*70)
print(" CARREGAMENTO DIRETO DO EMOTIC (SEM CONVERSÃO)")
print("="*70)
print()

# Detectar diretório do projeto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

print(f" Diretório atual: {current_dir}")
print(f" Diretório do projeto: {project_root}")

# Caminhos
EMOTIC_ROOT = os.path.join(project_root, "data", "emotic")
IMG_ARRS_DIR = os.path.join(EMOTIC_ROOT, "img_arrs")

print(f" EMOTIC_ROOT: {EMOTIC_ROOT}")
print(f" IMG_ARRS_DIR: {IMG_ARRS_DIR}")
print(f" IMG_ARRS existe? {os.path.exists(IMG_ARRS_DIR)}")
print()

# Emoções
EMOTIONS = [
    'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
    'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
    'Sympathy', 'Doubt', 'Disconnection', 'Fatigue', 'Embarrassment',
    'Yearning', 'Disapproval', 'Aversion', 'Annoyance', 'Anger',
    'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain', 'Suffering'
]

def process_split(csv_path, split_name):
    """Processa um split (train/val/test)"""

    if not os.path.exists(csv_path):
        print(f" Não encontrado: {csv_path}")
        return []

    print(f"\n Processando {split_name}...")
    df = pd.read_csv(csv_path)
    print(f"  Total: {len(df)} amostras")

    # DEBUG: mostrar primeira linha
    if len(df) > 0:
        print(f"\n   DEBUG - Primeira linha:")
        first_row = df.iloc[0]
        print(f"    Crop_name: '{first_row.get('Crop_name', 'N/A')}'")
        print(f"    Arr_name: '{first_row.get('Arr_name', 'N/A')}'")

        # Verificar se Crop_name existe
        crop_test = str(first_row.get('Crop_name', ''))
        full_path = os.path.join(IMG_ARRS_DIR, crop_test)
        print(f"    Caminho construído: {full_path}")
        print(f"    Existe? {os.path.exists(full_path)}")

        # Listar o que está em img_arrs
        if os.path.exists(IMG_ARRS_DIR):
            files = os.listdir(IMG_ARRS_DIR)
            print(f"\n   Ficheiros em img_arrs: {len(files)} ficheiros")
            if len(files) > 0:
                print(f"    Primeiros 10: {files[:10]}")

                # Verificar se crop_arr_train_0.npy existe
                if 'crop_arr_train_0.npy' in files:
                    print(f"     crop_arr_train_0.npy EXISTE!")
                else:
                    print(f"     crop_arr_train_0.npy NÃO EXISTE")

                # Procurar padrões
                train_files = [f for f in files if 'train' in f.lower()]
                val_files = [f for f in files if 'val' in f.lower()]
                test_files = [f for f in files if 'test' in f.lower()]

                print(f"\n   Ficheiros por tipo:")
                print(f"    Train: {len(train_files)} (exemplos: {train_files[:3]})")
                print(f"    Val: {len(val_files)} (exemplos: {val_files[:3]})")
                print(f"    Test: {len(test_files)} (exemplos: {test_files[:3]})")
        else:
            print(f"   Pasta não existe: {IMG_ARRS_DIR}")

    annotations = []
    found = 0
    missing = 0

    for idx, row in df.iterrows():
        # Obter caminho do .npy
        crop_name = str(row.get('Crop_name', ''))
        npy_path = os.path.join(IMG_ARRS_DIR, crop_name)

        if not os.path.exists(npy_path):
            missing += 1
            continue

        # Extrair emoção principal
        emotion = None
        for emo in EMOTIONS:
            col_name = emo.replace('/', '')  # Doubt/Confusion -> Doubt
            if col_name in df.columns:
                try:
                    if int(row[col_name]) == 1:
                        emotion = emo
                        break
                except:
                    pass

        if emotion is None:
            emotion = np.random.choice(EMOTIONS)

        # Criar anotação (usar caminho .npy diretamente!)
        annotation = {
            'image_path': npy_path,  # Caminho COMPLETO do .npy
            'emotion': emotion,
            'bbox': [0, 0, int(row.get('Width', 224)), int(row.get('Height', 224))]
        }

        annotations.append(annotation)
        found += 1

    print(f"   Encontrados: {found}")
    if missing > 0:
        print(f"   Faltando: {missing}")

    return annotations

# Processar todos os splits
all_annotations = []

csv_files = [
    (os.path.join(EMOTIC_ROOT, "annot_arrs_train.csv"), "train"),
    (os.path.join(EMOTIC_ROOT, "annot_arrs_val.csv"), "val"),
    (os.path.join(EMOTIC_ROOT, "annot_arrs_test.csv"), "test"),
]

for csv_path, split_name in csv_files:
    annotations = process_split(csv_path, split_name)
    all_annotations.extend(annotations)

# Salvar
if all_annotations:
    output_json = os.path.join(EMOTIC_ROOT, "annotations.json")
    with open(output_json, 'w') as f:
        json.dump(all_annotations, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  CARREGAMENTO CONCLUÍDO")
    print(f"{'='*70}")
    print(f"\n Total de anotações: {len(all_annotations)}")
    print(f" Ficheiro criado: {output_json}")

    # Estatísticas
    emotions_count = {}
    for ann in all_annotations:
        emotion = ann['emotion']
        emotions_count[emotion] = emotions_count.get(emotion, 0) + 1

    print(f"\n Distribuição de emoções (top 10):")
    sorted_emotions = sorted(emotions_count.items(), key=lambda x: x[1], reverse=True)
    for emotion, count in sorted_emotions[:10]:
        print(f"  {emotion}: {count}")

    print(f"\n Agora pode executar: python train.py")
    print(f"\n NOTA: O data_loader foi atualizado para ler .npy diretamente!")

else:
    print("\n Nenhuma anotação processada!")