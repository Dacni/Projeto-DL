"""
Demonstração visual da remoção de rostos
Mostra imagens ANTES e DEPOIS para usar na apresentação
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from face_removal import FaceRemover
import random

print("="*70)
print(" DEMONSTRAÇÃO DE REMOÇÃO DE ROSTOS")
print("="*70)
print()


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

ANNOTATIONS_FILE = "data/emotic/annotations.json"
OUTPUT_DIR = "results/face_removal_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar anotações
print("Carregando anotações...")
with open(ANNOTATIONS_FILE, 'r') as f:
    annotations = json.load(f)

print(f" Total de imagens: {len(annotations)}")

# Inicializar removedor de rostos
print("\nInicializando FaceRemover...")
face_remover = FaceRemover(blur_intensity=51)
print(" FaceRemover pronto")
print()

# ============================================================================
# SELECIONAR AMOSTRAS
# ============================================================================

# Pegar 6 amostras aleatórias
num_samples = 6
samples = random.sample(annotations, num_samples)

print(f"Processando {num_samples} amostras aleatórias...\n")

# ============================================================================
# PROCESSAR E VISUALIZAR
# ============================================================================

fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 3))

for idx, ann in enumerate(samples):
    try:
        # Carregar imagem original
        img_original = np.load(ann['image_path'], allow_pickle=True)
        
        # Normalizar se necessário
        if img_original.dtype != np.uint8:
            if img_original.max() <= 1.0:
                img_original = (img_original * 255).astype(np.uint8)
            else:
                img_original = np.clip(img_original, 0, 255).astype(np.uint8)
        
        # Converter para BGR (FaceRemover espera BGR)
        if len(img_original.shape) == 3:
            img_bgr = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_original
        
        # Remover rosto
        img_processed = face_remover.remove_face(img_bgr.copy())
        
        # Converter de volta para RGB para visualização
        img_processed_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        
        # ANTES
        axes[idx, 0].imshow(img_original)
        axes[idx, 0].set_title(f'ANTES (Imagem {idx+1})\nEmoção: {ann["emotion"]}', 
                              fontsize=11, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # DEPOIS
        axes[idx, 1].imshow(img_processed_rgb)
        axes[idx, 1].set_title(f'DEPOIS (Rosto Removido)\nEmoção: {ann["emotion"]}', 
                              fontsize=11, fontweight='bold')
        axes[idx, 1].axis('off')
        
        print(f" Processada amostra {idx+1}: {ann['emotion']}")
        
    except Exception as e:
        print(f" Erro na amostra {idx+1}: {e}")
        axes[idx, 0].text(0.5, 0.5, 'Erro ao carregar', ha='center', va='center')
        axes[idx, 0].axis('off')
        axes[idx, 1].text(0.5, 0.5, 'Erro ao processar', ha='center', va='center')
        axes[idx, 1].axis('off')

plt.suptitle('Demonstração de Remoção de Rostos - ANTES vs DEPOIS', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Salvar
output_path = os.path.join(OUTPUT_DIR, 'face_removal_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n Salvo: {output_path}")

plt.show()

# ============================================================================
# CRIAR VISUALIZAÇÃO GRANDE (1 EXEMPLO)
# ============================================================================

print("\nCriando visualização ampliada de 1 exemplo...")

# Pegar um exemplo com pessoa de frente
sample = random.choice(annotations)

try:
    # Carregar e processar
    img_original = np.load(sample['image_path'], allow_pickle=True)
    
    if img_original.dtype != np.uint8:
        if img_original.max() <= 1.0:
            img_original = (img_original * 255).astype(np.uint8)
        else:
            img_original = np.clip(img_original, 0, 255).astype(np.uint8)
    
    img_bgr = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    img_processed = face_remover.remove_face(img_bgr.copy())
    img_processed_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    
    # Criar visualização lado-a-lado GRANDE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ANTES
    ax1.imshow(img_original)
    ax1.set_title('ANTES\n(Com Rosto Visível)', fontsize=18, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # DEPOIS
    ax2.imshow(img_processed_rgb)
    ax2.set_title('DEPOIS\n(Rosto Removido com Gaussian Blur)', fontsize=18, fontweight='bold', pad=20)
    ax2.axis('off')
    
    # Seta no meio
    fig.text(0.5, 0.5, '', fontsize=60, ha='center', va='center', 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle(f'Remoção de Rostos - Emoção: {sample["emotion"]}', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path_large = os.path.join(OUTPUT_DIR, 'face_removal_large_example.png')
    plt.savefig(output_path_large, dpi=300, bbox_inches='tight')
    print(f" Salvo: {output_path_large}")
    
    plt.show()

except Exception as e:
    print(f" Erro ao criar exemplo ampliado: {e}")

# ============================================================================
# CRIAR SLIDE PARA APRESENTAÇÃO
# ============================================================================

print("\nCriando slide otimizado para apresentação...")

# Selecionar 3 exemplos com emoções diferentes
emotions_wanted = ['Happiness', 'Engagement', 'Peace']
slide_samples = []

for emotion in emotions_wanted:
    emotion_anns = [a for a in annotations if a['emotion'] == emotion]
    if emotion_anns:
        slide_samples.append(random.choice(emotion_anns))

if len(slide_samples) < 3:
    # Fallback: pegar 3 aleatórias
    slide_samples = random.sample(annotations, 3)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for idx, ann in enumerate(slide_samples):
    try:
        # Processar
        img_original = np.load(ann['image_path'], allow_pickle=True)
        
        if img_original.dtype != np.uint8:
            if img_original.max() <= 1.0:
                img_original = (img_original * 255).astype(np.uint8)
            else:
                img_original = np.clip(img_original, 0, 255).astype(np.uint8)
        
        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        img_processed = face_remover.remove_face(img_bgr.copy())
        img_processed_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        
        # Exibir
        axes[idx, 0].imshow(img_original)
        axes[idx, 0].set_title('Original', fontsize=14, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img_processed_rgb)
        axes[idx, 1].set_title('Rosto Removido', fontsize=14, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Label da emoção à esquerda
        fig.text(0.01, 0.83 - (idx * 0.31), f'{ann["emotion"]}', 
                fontsize=13, fontweight='bold', rotation=90, va='center')
        
    except Exception as e:
        print(f" Erro na amostra {idx}: {e}")

plt.suptitle('Técnica de Remoção de Rostos\nMediaPipe Face Detection + Gaussian Blur', 
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0.03, 0, 1, 0.96])

output_slide = os.path.join(OUTPUT_DIR, 'slide_face_removal.png')
plt.savefig(output_slide, dpi=300, bbox_inches='tight')
print(f" Salvo slide: {output_slide}")

plt.show()

# ============================================================================
# RESUMO
# ============================================================================

print("\n" + "="*70)
print(" DEMONSTRAÇÃO CONCLUÍDA")
print("="*70)
print("\n Ficheiros criados em results/face_removal_demo/:")
print("   1. face_removal_comparison.png    - 6 exemplos antes/depois")
print("   2. face_removal_large_example.png - 1 exemplo ampliado")
print("   3. slide_face_removal.png         - Slide para apresentação")
print("\n Use 'slide_face_removal.png' no Slide 7 da apresentação!")
print()
