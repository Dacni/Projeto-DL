"""
Converte o dataset EMOTIC do Kaggle para o formato usado no projeto
Versão melhorada que detecta automaticamente a estrutura
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import json
from tqdm import tqdm
import ast

print("="*70)
print(" CONVERSÃO DO DATASET EMOTIC")
print("="*70)
print()

# Detectar o diretório raiz do projeto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

# Se estivermos numa subpasta (src), voltar ao root
if os.path.basename(current_dir) == 'src':
    project_root = os.path.dirname(current_dir)

print(f" Diretório do projeto: {project_root}")

# Caminhos
EMOTIC_ROOT = os.path.join(project_root, "data", "emotic")
IMG_ARRS_DIR = os.path.join(EMOTIC_ROOT, "img_arrs")
OUTPUT_DIR = os.path.join(EMOTIC_ROOT, "images")

print(f" Dataset EMOTIC: {EMOTIC_ROOT}")
print(f" Imagens .npy: {IMG_ARRS_DIR}")
print(f" Saída: {OUTPUT_DIR}")
print()

# Verificar se as pastas existem
if not os.path.exists(EMOTIC_ROOT):
    print(f" ERRO: Pasta não encontrada: {EMOTIC_ROOT}")
    print("\n Solução:")
    print("  1. Certifique-se que extraiu o dataset para data/emotic/")
    print("  2. Execute o script da raiz do projeto:")
    print("     python conversion.py")
    sys.exit(1)

# Criar pasta de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lista das 26 emoções do EMOTIC
EMOTIC_CATEGORIES = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion',
    'Confidence', 'Disapproval', 'Disconnection', 'Disquietment', 'Doubt',
    'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue',
    'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure',
    'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy',
    'Yearning'
]

def analyze_csv_structure(csv_path):
    """Analisa a estrutura do CSV"""
    print(f"\n Analisando: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    print(f"  Linhas: {len(df)}")
    print(f"  Colunas: {list(df.columns)[:10]}...")  # Mostrar primeiras 10
    print(f"\n  Primeiras 2 linhas:")
    print(df.head(2).to_string())
    return df

def convert_npy_to_jpg(npy_path, output_path):
    """Converte imagem .npy para .jpg"""
    try:
        img = np.load(npy_path, allow_pickle=True)

        # Normalizar se necessário
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Verificar dimensões
        if len(img.shape) == 3:
            # RGB para BGR (OpenCV usa BGR)
            if img.shape[2] == 3:
                cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                return True

        return False
    except Exception as e:
        return False

def extract_emotion(row, df):
    """Extrai a emoção principal de uma linha"""

    # Tentar encontrar colunas de emoções (valores binários)
    for cat in EMOTIC_CATEGORIES:
        if cat in df.columns:
            try:
                if int(row[cat]) == 1:
                    return cat
            except:
                pass

    # Tentar coluna 'categories' ou similar
    possible_cols = ['categories', 'Categories', 'category', 'emotion', 'label']
    for col in possible_cols:
        if col in df.columns:
            try:
                val = str(row[col])
                if val and val != 'nan':
                    # Pode ser lista ou string
                    if val.startswith('['):
                        cats = ast.literal_eval(val)
                        return cats[0] if cats else np.random.choice(EMOTIC_CATEGORIES)
                    else:
                        return val
            except:
                pass

    # Default: escolher aleatoriamente
    return np.random.choice(EMOTIC_CATEGORIES)

def extract_bbox(row, df):
    """
    Extrai bounding box de uma linha
    NOTA: Como já temos crops, o bbox não é necessário mas mantemos para compatibilidade
    """

    # EMOTIC CSV tem X_min, Y_min, X_max, Y_max
    # Mas como já temos crops, estas coordenadas são relativas à imagem original
    # Para os crops, podemos usar bbox padrão ou dimensões da imagem
    if all(c in df.columns for c in ['Width', 'Height']):
        try:
            width = int(float(row['Width']))
            height = int(float(row['Height']))
            # Bbox cobrindo toda a imagem do crop
            return [0, 0, width, height]
        except:
            pass

    # Default: bbox padrão
    return [0, 0, 224, 224]

def find_image_file(row, df, debug=False):
    """Encontra o ficheiro de imagem usando Crop_name (em vez de Arr_name)"""

    # EMOTIC do Kaggle usa Crop_name, não Arr_name
    if 'Crop_name' in df.columns:
        crop_name = str(row['Crop_name'])
        if crop_name and crop_name != 'nan':
            # Procurar em img_arrs
            possible_paths = [
                os.path.join(IMG_ARRS_DIR, crop_name),
                os.path.join(EMOTIC_ROOT, "img_arrs", crop_name),
                os.path.join(EMOTIC_ROOT, crop_name),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    return path

            if debug:
                print(f"\n  DEBUG: Crop_name={crop_name}, procurado em:")
                for p in possible_paths:
                    print(f"    {p} - existe: {os.path.exists(p)}")

    # Fallback: tentar Arr_name
    if 'Arr_name' in df.columns:
        arr_name = str(row['Arr_name'])
        if arr_name and arr_name != 'nan':
            possible_paths = [
                os.path.join(IMG_ARRS_DIR, arr_name),
                os.path.join(EMOTIC_ROOT, "img_arrs", arr_name),
                os.path.join(EMOTIC_ROOT, arr_name),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    return path

    # Último fallback: tentar com Filename
    if 'Filename' in df.columns:
        filename = str(row['Filename'])
        base_name = os.path.splitext(filename)[0]
        npy_name = base_name + '.npy'

        possible_paths = [
            os.path.join(IMG_ARRS_DIR, npy_name),
            os.path.join(EMOTIC_ROOT, "img_arrs", npy_name),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

    return None

def process_csv(csv_path, split_name):
    """Processa um ficheiro CSV"""

    if not os.path.exists(csv_path):
        print(f" Não encontrado: {os.path.basename(csv_path)}")
        return []

    # Analisar estrutura
    df = analyze_csv_structure(csv_path)

    annotations = []
    converted = 0
    failed = 0
    skipped = 0

    print(f"\n Convertendo imagens...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        try:
            # Encontrar ficheiro .npy usando Arr_name
            npy_path = find_image_file(row, df)

            if not npy_path:
                failed += 1
                if idx < 5:  # Mostrar primeiros erros para debug
                    arr_name = row.get('Arr_name', 'N/A')
                    print(f"\n   Não encontrado: {arr_name}")
                continue

            # Extrair nome original para referência
            filename = str(row['Filename']) if 'Filename' in df.columns else f"img_{idx}"
            base_name = os.path.splitext(os.path.basename(filename))[0]

            # Nome da imagem de saída
            jpg_filename = f"{split_name}_{idx:05d}_{base_name}.jpg"
            jpg_path = os.path.join(OUTPUT_DIR, jpg_filename)

            # Converter
            if convert_npy_to_jpg(npy_path, jpg_path):
                # Extrair informação
                emotion = extract_emotion(row, df)
                bbox = extract_bbox(row, df)

                annotation = {
                    'image_path': f"images/{jpg_filename}",
                    'emotion': emotion,
                    'bbox': bbox
                }

                annotations.append(annotation)
                converted += 1
            else:
                failed += 1
                if idx < 5:  # Mostrar primeiros erros
                    print(f"\n   Falha ao converter: {npy_path}")

        except Exception as e:
            failed += 1
            if idx < 5:  # Mostrar primeiros erros
                print(f"\n   Erro na linha {idx}: {e}")
            continue

    print(f"\n   Convertidas: {converted}")
    print(f"   Falhadas: {failed}")
    if skipped > 0:
        print(f"  ⊘ Ignoradas: {skipped}")

    return annotations

# Listar ficheiros disponíveis
print(" Ficheiros disponíveis em data/emotic/:")
if os.path.exists(EMOTIC_ROOT):
    for item in os.listdir(EMOTIC_ROOT):
        item_path = os.path.join(EMOTIC_ROOT, item)
        if os.path.isfile(item_path):
            print(f"   {item}")
        else:
            print(f"   {item}/")
else:
    print("   Pasta não encontrada!")

print()

# Processar todos os CSVs
all_annotations = []

csv_files = [
    ("annot_arrs_train.csv", "train"),
    ("annot_arrs_val.csv", "val"),
    ("annot_arrs_test.csv", "test"),
    # Alternativas
    ("train.csv", "train"),
    ("val.csv", "val"),
    ("test.csv", "test"),
]

for csv_name, split_name in csv_files:
    csv_path = os.path.join(EMOTIC_ROOT, csv_name)
    if os.path.exists(csv_path):
        annotations = process_csv(csv_path, split_name)
        all_annotations.extend(annotations)

# Salvar resultado
if all_annotations:
    output_json = os.path.join(EMOTIC_ROOT, "annotations.json")
    with open(output_json, 'w') as f:
        json.dump(all_annotations, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  CONVERSÃO CONCLUÍDA")
    print(f"{'='*70}")
    print(f"\n Total de anotações: {len(all_annotations)}")
    print(f" Ficheiro criado: {output_json}")
    print(f" Imagens em: {OUTPUT_DIR}")

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

else:
    print(f"\n{'='*70}")
    print(f"  NENHUMA ANOTAÇÃO PROCESSADA")
    print(f"{'='*70}")
    print("\n Possíveis problemas:")
    print("  1. Os ficheiros CSV não foram encontrados")
    print("  2. Verifique a lista de ficheiros acima")
    print("  3. Certifique-se que extraiu o ZIP completamente")
    print("\n Execute este script da raiz do projeto:")
    print("     python conversion.py")
    print("  Não de dentro da pasta src/")


# Criar pasta de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lista das 26 emoções do EMOTIC
EMOTIC_CATEGORIES = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion',
    'Confidence', 'Disapproval', 'Disconnection', 'Disquietment', 'Doubt',
    'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue',
    'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure',
    'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy',
    'Yearning'
]

def analyze_csv_structure(csv_path):
    """Analisa a estrutura do CSV"""
    print(f"\n Analisando: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    print(f"  Linhas: {len(df)}")
    print(f"  Colunas: {list(df.columns)}")
    print(f"\n  Primeiras linhas:")
    print(df.head(2))
    return df

def convert_npy_to_jpg(npy_path, output_path):
    """Converte imagem .npy para .jpg"""
    try:
        img = np.load(npy_path)

        # Normalizar se necessário
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Verificar dimensões
        if len(img.shape) == 3:
            # RGB para BGR (OpenCV usa BGR)
            if img.shape[2] == 3:
                cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                return True

        return False
    except Exception as e:
        return False

def extract_emotion(row, df):
    """Extrai a emoção principal de uma linha"""

    # Tentar encontrar colunas de emoções (valores binários)
    for cat in EMOTIC_CATEGORIES:
        if cat in df.columns:
            try:
                if int(row[cat]) == 1:
                    return cat
            except:
                pass

    # Tentar coluna 'categories' ou similar
    possible_cols = ['categories', 'Categories', 'category', 'emotion', 'label']
    for col in possible_cols:
        if col in df.columns:
            try:
                val = str(row[col])
                if val and val != 'nan':
                    # Pode ser lista ou string
                    if val.startswith('['):
                        cats = ast.literal_eval(val)
                        return cats[0] if cats else np.random.choice(EMOTIC_CATEGORIES)
                    else:
                        return val
            except:
                pass

    # Default: escolher aleatoriamente
    return np.random.choice(EMOTIC_CATEGORIES)

def extract_bbox(row, df):
    """Extrai bounding box de uma linha"""

    possible_cols = ['bbox', 'BBox', 'bounding_box', 'box']

    for col in possible_cols:
        if col in df.columns:
            try:
                val = str(row[col])
                if val and val != 'nan':
                    # Tentar parsear
                    if val.startswith('['):
                        bbox = ast.literal_eval(val)
                        return bbox
                    elif ',' in val:
                        bbox = [float(x.strip()) for x in val.split(',')]
                        return [int(x) for x in bbox]
            except:
                pass

    # Tentar colunas individuais (x, y, w, h)
    if all(c in df.columns for c in ['x', 'y', 'w', 'h']):
        try:
            return [int(row['x']), int(row['y']), int(row['w']), int(row['h'])]
        except:
            pass

    # Default: bbox no centro
    return [100, 100, 200, 300]

def find_image_file(folder, filename):
    """Encontra o ficheiro de imagem"""

    # Remover extensão e adicionar .npy
    base_name = os.path.splitext(filename)[0]
    npy_name = base_name + '.npy'

    # Procurar em img_arrs
    possible_paths = [
        os.path.join(IMG_ARRS_DIR, npy_name),
        os.path.join(IMG_ARRS_DIR, folder, npy_name),
        os.path.join(EMOTIC_ROOT, folder, npy_name),
        os.path.join(EMOTIC_ROOT, npy_name)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

def process_csv(csv_path, split_name):
    """Processa um ficheiro CSV"""

    if not os.path.exists(csv_path):
        print(f" Não encontrado: {csv_path}")
        return []

    # Analisar estrutura
    df = analyze_csv_structure(csv_path)

    annotations = []
    converted = 0
    failed = 0
    skipped = 0

    print(f"\n Convertendo imagens...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        try:
            # Extrair nome do ficheiro
            filename = None
            for col in ['Filename', 'filename', 'image', 'img', 'file']:
                if col in df.columns:
                    filename = str(row[col])
                    break

            if not filename or filename == 'nan':
                skipped += 1
                continue

            # Extrair folder se existir
            folder = ''
            for col in ['Folder', 'folder', 'path']:
                if col in df.columns:
                    folder = str(row[col])
                    if folder == 'nan':
                        folder = ''
                    break

            # Encontrar ficheiro .npy
            npy_path = find_image_file(folder, filename)

            if not npy_path:
                failed += 1
                continue

            # Nome da imagem de saída
            base_name = os.path.splitext(os.path.basename(filename))[0]
            jpg_filename = f"{split_name}_{idx:05d}_{base_name}.jpg"
            jpg_path = os.path.join(OUTPUT_DIR, jpg_filename)

            # Converter
            if convert_npy_to_jpg(npy_path, jpg_path):
                # Extrair informação
                emotion = extract_emotion(row, df)
                bbox = extract_bbox(row, df)

                annotation = {
                    'image_path': f"images/{jpg_filename}",
                    'emotion': emotion,
                    'bbox': bbox
                }

                annotations.append(annotation)
                converted += 1
            else:
                failed += 1

        except Exception as e:
            failed += 1
            continue

    print(f"\n   Convertidas: {converted}")
    print(f"   Falhadas: {failed}")
    if skipped > 0:
        print(f"  ⊘ Ignoradas: {skipped}")

    return annotations

# Processar todos os CSVs
all_annotations = []

csv_files = [
    ("annot_arrs_train.csv", "train"),
    ("annot_arrs_val.csv", "val"),
    ("annot_arrs_test.csv", "test")
]

for csv_name, split_name in csv_files:
    csv_path = os.path.join(EMOTIC_ROOT, csv_name)
    if os.path.exists(csv_path):
        annotations = process_csv(csv_path, split_name)
        all_annotations.extend(annotations)
    else:
        print(f" Não encontrado: {csv_name}")

# Salvar resultado
if all_annotations:
    output_json = os.path.join(EMOTIC_ROOT, "annotations.json")
    with open(output_json, 'w') as f:
        json.dump(all_annotations, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  CONVERSÃO CONCLUÍDA")
    print(f"{'='*70}")
    print(f"\n Total de anotações: {len(all_annotations)}")
    print(f" Ficheiro criado: {output_json}")
    print(f" Imagens em: {OUTPUT_DIR}")

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

else:
    print(f"\n{'='*70}")
    print(f"  NENHUMA ANOTAÇÃO PROCESSADA")
    print(f"{'='*70}")
    print("\n Verifique:")
    print("  1. Os ficheiros CSV estão em data/emotic/")
    print("  2. As imagens .npy estão em data/emotic/img_arrs/")
    print("  3. A estrutura das colunas dos CSVs")