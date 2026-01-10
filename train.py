"""
Script de treino completo do projeto
Reconhecimento de Emoções através de Posturas Corporais
"""

import os
import sys

# Garantir que pode importar os módulos locais
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=" * 70)
print(" RECONHECIMENTO DE EMOÇÕES - TREINO COMPLETO")
print(" Universidade Católica Portuguesa")
print(" Autor: Daniel Rodrigues")
print("=" * 70)
print()

# Verificar instalação
print("Verificando instalação...")
try:
    import tensorflow as tf

    print(f" TensorFlow {tf.__version__}")

    import cv2

    print(f" OpenCV {cv2.__version__}")

    import numpy as np

    print(f" NumPy {np.__version__}")

    import pandas as pd

    print(f" Pandas {pd.__version__}")

    # Verificar GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f" GPU detectada: {len(gpus)} dispositivo(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print(" GPU não detectada - usando CPU (será mais lento)")

    print()

except ImportError as e:
    print(f"\n ERRO: Biblioteca não instalada: {e}")
    print("\nExecute: pip install -r requirements.txt")
    sys.exit(1)

# Imports do projeto
print("Importando módulos do projeto...")

try:
    from config import CONFIG

    print(" config")
except ImportError as e:
    print(f" Erro: {e}")
    sys.exit(1)

try:
    from data_loader import EMOTICDataLoader, EmotionDataGenerator

    print(" data_loader")
except ImportError as e:
    print(f" Erro: {e}")
    sys.exit(1)

try:
    from cnn_models import EmotionCNNModels, create_callbacks

    print(" cnn_models")
except ImportError as e:
    print(f" Erro: {e}")
    sys.exit(1)

try:
    from evaluation import ModelEvaluator

    print(" evaluation")
except ImportError as e:
    print(f" Erro: {e}")
    sys.exit(1)

import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("\n Todos os módulos importados com sucesso!\n")


# ============================================================================
# FUNÇÃO PRINCIPAL DE TREINO
# ============================================================================

def train_model():
    """Função principal de treino"""

    # Passo 1: Verificar annotations.json
    print("=" * 70)
    print(" PASSO 1: Carregar Anotações")
    print("=" * 70)
    print()

    annotations_file = "data/emotic/annotations.json"

    if not os.path.exists(annotations_file):
        print(f" ERRO: {annotations_file} não encontrado!")
        print("\nExecute primeiro: python load_emotic_direct.py")
        sys.exit(1)

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    print(f" Carregadas {len(annotations)} anotações")
    print()

    # Passo 2: Preparar dados
    print("=" * 70)
    print(" PASSO 2: Preparar Dados")
    print("=" * 70)
    print()

    # Extrair caminhos e emoções
    image_paths = [ann['image_path'] for ann in annotations]
    emotions = [ann['emotion'] for ann in annotations]

    # Codificar emoções
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(emotions)

    print(f" Classes de emoções: {len(label_encoder.classes_)}")
    print(f"  Exemplos: {list(label_encoder.classes_[:10])}")
    print()

    # Atualizar CONFIG com número correto de classes
    CONFIG['num_classes'] = len(label_encoder.classes_)

    # Split treino/validação/teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths,
        y_encoded,
        test_size=CONFIG['test_split'],
        random_state=42,
        stratify=y_encoded
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=CONFIG['val_split'] / (CONFIG['train_split'] + CONFIG['val_split']),
        random_state=42,
        stratify=y_temp
    )

    print(f" Divisão dos dados:")
    print(f"  Treino: {len(X_train)} ({len(X_train) / len(image_paths) * 100:.1f}%)")
    print(f"  Validação: {len(X_val)} ({len(X_val) / len(image_paths) * 100:.1f}%)")
    print(f"  Teste: {len(X_test)} ({len(X_test) / len(image_paths) * 100:.1f}%)")
    print()

    # Passo 3: Criar geradores de dados
    print("=" * 70)
    print(" PASSO 3: Criar Geradores de Dados")
    print("=" * 70)
    print()

    img_size = (CONFIG['img_height'], CONFIG['img_width'])

    train_gen = EmotionDataGenerator(
        np.array(X_train),
        y_train,
        batch_size=CONFIG['batch_size'],
        img_size=img_size,
        augment=True  # Data augmentation no treino
    )

    val_gen = EmotionDataGenerator(
        np.array(X_val),
        y_val,
        batch_size=CONFIG['batch_size'],
        img_size=img_size,
        augment=False
    )

    test_gen = EmotionDataGenerator(
        np.array(X_test),
        y_test,
        batch_size=CONFIG['batch_size'],
        img_size=img_size,
        augment=False
    )

    print(f" Geradores criados:")
    print(f"  Treino: {len(train_gen)} batches")
    print(f"  Validação: {len(val_gen)} batches")
    print(f"  Teste: {len(test_gen)} batches")
    print()

    # Passo 4: Criar e treinar modelo
    print("=" * 70)
    print(" PASSO 4: Criar e Treinar Modelo")
    print("=" * 70)
    print()

    # Criar diretórios
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Criar modelo (usando ResNet50 - melhor performance)
    print("Criando modelo ResNet50 com Transfer Learning...")
    input_shape = (CONFIG['img_height'], CONFIG['img_width'], 3)

    model = EmotionCNNModels.create_resnet_transfer(
        input_shape,
        CONFIG['num_classes']
    )

    # Compilar
    model = EmotionCNNModels.compile_model(model, CONFIG['learning_rate'])

    print(f" Modelo criado com {model.count_params():,} parâmetros")
    print()

    # Callbacks
    callbacks = create_callbacks('resnet50_emotion')

    # Treinar
    print("=" * 70)
    print(" INICIANDO TREINO")
    print("=" * 70)
    print(f"\nÉpocas: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print()
    print("⏳ Isto pode demorar várias horas...")
    print()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    print("\n Treino concluído!")
    print()

    # Passo 5: Avaliar modelo
    print("=" * 70)
    print(" PASSO 5: Avaliar Modelo")
    print("=" * 70)
    print()

    # Fazer predições no conjunto de teste
    print("Fazendo predições no conjunto de teste...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calcular métricas
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\n RESULTADOS FINAIS:")
    print(f"{'=' * 70}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  F1-Score (Macro): {f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"{'=' * 70}")
    print()

    # Salvar relatório
    evaluator = ModelEvaluator(model, label_encoder)
    evaluator.plot_confusion_matrix(
        y_test,
        y_pred,
        save_path='results/confusion_matrix.png'
    )

    evaluator.classification_report_detailed(
        y_test,
        y_pred,
        save_path='results/classification_report.txt'
    )

    # Salvar modelo final
    print("Salvando modelo final...")
    model.save('models/emotion_resnet50_final.h5')

    # Salvar label encoder
    import joblib
    joblib.dump(label_encoder, 'models/label_encoder.pkl')

    print(" Modelo salvo: models/emotion_resnet50_final.h5")
    print(" Label encoder salvo: models/label_encoder.pkl")
    print()

    # Passo 6: Plotar histórico de treino
    print("Gerando gráficos de treino...")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Treino')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validação')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Treino')
    axes[0, 1].plot(history.history['val_loss'], label='Validação')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Treino')
        axes[1, 0].plot(history.history['val_precision'], label='Validação')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Treino')
        axes[1, 1].plot(history.history['val_recall'], label='Validação')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    print(" Gráficos salvos: results/training_history.png")
    print()

    # Resumo final
    print("=" * 70)
    print(" TREINO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    print()
    print(" Ficheiros gerados:")
    print("  - models/emotion_resnet50_final.h5")
    print("  - models/label_encoder.pkl")
    print("  - results/confusion_matrix.png")
    print("  - results/classification_report.txt")
    print("  - results/training_history.png")
    print()
    print(" Projeto de Aprendizagem Profunda - UCP")
    print("   Reconhecimento de Emoções através de Posturas Corporais")
    print("=" * 70)


# ============================================================================
# EXECUTAR
# ============================================================================

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\n Treino interrompido pelo utilizador")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n ERRO durante treino: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)