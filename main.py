"""
Pipeline principal do projeto
Reconhecimento de Emoções através de Posturas Corporais
"""

import os
import sys

# Adicionar diretório atual ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Imports necessários
import tensorflow as tf

# Importar módulos do projeto
print("Importando módulos do projeto...")

try:
    from config import CONFIG

    print(" config")
except ImportError as e:
    print(f" Erro ao importar config: {e}")
    sys.exit(1)

try:
    from face_removal import FaceRemover

    print(" face_removal")
except ImportError as e:
    print(f" Erro ao importar face_removal: {e}")
    sys.exit(1)

try:
    from data_loader import EMOTICDataLoader

    print(" data_loader")
except ImportError as e:
    print(f" Erro ao importar data_loader: {e}")
    sys.exit(1)

try:
    from training import train_multiple_models

    print(" training")
except ImportError as e:
    print(f" Erro ao importar training: {e}")
    sys.exit(1)

try:
    from evaluation import compare_models, ModelEvaluator

    print(" evaluation")
except ImportError as e:
    print(f" Erro ao importar evaluation: {e}")
    sys.exit(1)

print("\n Todos os módulos importados com sucesso!\n")


def main():
    """Pipeline principal de execução"""

    print("=" * 70)
    print(" RECONHECIMENTO DE EMOÇÕES ATRAVÉS DE POSTURAS CORPORAIS")
    print(" Universidade Católica Portuguesa - Aprendizagem Profunda")
    print(" Autor: Daniel Rodrigues")
    print("=" * 70 + "\n")

    # ========== PASSO 1: Configurar diretórios ==========
    print("PASSO 1: Configurando diretórios...")

    DATA_DIR = "data/emotic"
    ANNOTATIONS_FILE = "data/emotic/annotations.json"  # Ajuste conforme necessário
    CROPS_DIR = "data/crops"
    PROCESSED_DIR = "data/processed"

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(" Diretórios configurados\n")

    # ========== PASSO 2: Carregar dados ==========
    print("PASSO 2: Carregando dataset EMOTIC...")

    loader = EMOTICDataLoader(DATA_DIR, ANNOTATIONS_FILE)
    df = loader.load_annotations()

    # Verificar distribuição de emoções
    print("\nDistribuição de emoções:")
    print(loader.get_emotion_distribution())
    print()

    # ========== PASSO 3: Extrair crops das pessoas ==========
    print("PASSO 3: Extraindo crops individuais das pessoas...")

    crops_df = loader.extract_person_crops(CROPS_DIR)
    print()

    # ========== PASSO 4: Remover rostos ==========
    print("PASSO 4: Removendo rostos das imagens...")

    face_remover = FaceRemover(blur_intensity=50)
    face_remover.process_batch(
        crops_df['crop_path'].tolist(),
        PROCESSED_DIR
    )

    # Atualizar caminhos para versões processadas
    crops_df['processed_path'] = crops_df['crop_path'].apply(
        lambda x: os.path.join(PROCESSED_DIR, os.path.basename(x))
    )
    print()

    # ========== PASSO 5: Preparar dados para treino ==========
    print("PASSO 5: Preparando dados para treino...")

    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_data(crops_df)
    print()

    # ========== PASSO 6: Treinar modelos ==========
    print("PASSO 6: Treinando modelos...")
    print("Isto pode demorar várias horas dependendo do hardware...\n")

    trained_models = train_multiple_models(
        X_train, X_val, y_train, y_val, CONFIG
    )

    # Plotar históricos de treino
    for model_name, trainer in trained_models.items():
        trainer.plot_training_history(
            save_path=f'results/{model_name}_training_history.png'
        )

    print("\n Todos os modelos treinados com sucesso!\n")

    # ========== PASSO 7: Avaliar modelos ==========
    print("PASSO 7: Avaliando modelos no conjunto de teste...")

    comparison_df = compare_models(
        trained_models, X_test, y_test, loader.label_encoder
    )

    # ========== PASSO 8: Análise detalhada do melhor modelo ==========
    print("\nPASSO 8: Análise detalhada do melhor modelo...")

    best_model_name = comparison_df.iloc[0]['Model']
    best_trainer = trained_models[best_model_name]

    print(f"\nMelhor modelo: {best_model_name}")
    print(f"Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")

    evaluator = ModelEvaluator(best_trainer.model, loader.label_encoder)
    metrics = evaluator.evaluate(X_test, y_test)

    # Análise de erros
    print("\nAnalisando erros de classificação...")
    evaluator.analyze_errors(X_test, metrics['y_true'], metrics['y_pred'])

    # Top predições
    print("\nVisualizando predições com maior confiança...")
    evaluator.plot_top_predictions(X_test, metrics['y_pred_probs'])

    # ========== PASSO 9: Salvar modelo final ==========
    print("\nPASSO 9: Salvando modelo final...")

    best_trainer.save_model(f'models/{best_model_name}_final.h5')

    # Salvar label encoder
    import joblib
    joblib.dump(loader.label_encoder, 'models/label_encoder.pkl')
    print(" Label encoder salvo")

    # ========== RELATÓRIO FINAL ==========
    print("\n" + "=" * 70)
    print(" TREINO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    print(f"\nMelhor modelo: {best_model_name}")
    print(f"Accuracy final: {comparison_df.iloc[0]['Accuracy']:.4f}")
    print(f"F1-Score (Macro): {comparison_df.iloc[0]['F1-Score (Macro)']:.4f}")
    print(f"F1-Score (Weighted): {comparison_df.iloc[0]['F1-Score (Weighted)']:.4f}")
    print("\nResultados salvos em:")
    print("  - models/")
    print("  - results/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Verificar se GPU está disponível
    print("Verificando hardware...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f" GPU detectada: {gpus}")
        # Configurar crescimento de memória
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print(" Nenhuma GPU detectada. O treino será feito na CPU (mais lento).")

    # Executar pipeline
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Treino interrompido pelo utilizador")
    except Exception as e:
        print(f"\n\n Erro durante execução: {str(e)}")
        import traceback

        traceback.print_exc()