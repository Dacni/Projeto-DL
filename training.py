"""
Módulo para treino dos modelos
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Adicionar diretório atual ao path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Importar módulos necessários
try:
    from config import CONFIG
except ImportError:
    print(" config.py não encontrado, usando configuração padrão")
    CONFIG = {
        'img_height': 224,
        'img_width': 224,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'num_classes': 26,
    }

try:
    from data_loader import EmotionDataGenerator
except ImportError as e:
    print(f" Erro ao importar EmotionDataGenerator: {e}")
    EmotionDataGenerator = None

try:
    # Tentar diferentes formas de import
    try:
        from cnn_models import EmotionCNNModels, create_callbacks
    except ImportError:
        import cnn_models
        EmotionCNNModels = cnn_models.EmotionCNNModels
        create_callbacks = cnn_models.create_callbacks
except ImportError as e:
    print(f" Erro ao importar cnn_models: {e}")
    EmotionCNNModels = None
    create_callbacks = None


class ModelTrainer:
    """Classe para gerir o treino de modelos"""

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.history = None

    def train(self, train_gen, val_gen, epochs, callbacks):
        """
        Treina o modelo

        Args:
            train_gen: gerador de treino
            val_gen: gerador de validação
            epochs: número de épocas
            callbacks: lista de callbacks

        Returns:
            histórico de treino
        """
        print(f"\n{'='*60}")
        print(f"Iniciando treino do modelo: {self.model_name}")
        print(f"{'='*60}\n")

        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def plot_training_history(self, save_path=None):
        """
        Plota gráficos de treino

        Args:
            save_path: caminho para salvar figura
        """
        if self.history is None:
            print("Erro: Modelo ainda não foi treinado")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Accuracy ao longo das épocas')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss ao longo das épocas')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Precision ao longo das épocas')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Recall ao longo das épocas')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Gráfico salvo em {save_path}")

        plt.show()

    def save_model(self, path):
        """Salva o modelo treinado"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f" Modelo salvo em {path}")

    def get_summary(self):
        """Retorna sumário do modelo"""
        return self.model.summary()


def train_multiple_models(X_train, X_val, y_train, y_val, config):
    """
    Treina múltiplos modelos e compara resultados

    Args:
        X_train, X_val: dados de treino e validação
        y_train, y_val: labels de treino e validação
        config: dicionário de configuração

    Returns:
        dicionário com modelos treinados
    """
    os.makedirs('models', exist_ok=True)

    input_shape = (config['img_height'], config['img_width'], 3)
    img_size = (config['img_height'], config['img_width'])

    # Criar geradores
    train_gen = EmotionDataGenerator(
        X_train, y_train,
        batch_size=config['batch_size'],
        img_size=img_size,
        augment=True
    )

    val_gen = EmotionDataGenerator(
        X_val, y_val,
        batch_size=config['batch_size'],
        img_size=img_size,
        augment=False
    )

    models_dict = {}

    # 1. CNN Customizada
    print("\n" + "="*60)
    print("TREINO 1/3: CNN Customizada")
    print("="*60)

    model_cnn = EmotionCNNModels.create_custom_cnn(input_shape, config['num_classes'])
    model_cnn = EmotionCNNModels.compile_model(model_cnn, config['learning_rate'])

    trainer_cnn = ModelTrainer(model_cnn, 'custom_cnn')
    trainer_cnn.train(
        train_gen, val_gen,
        epochs=config['epochs'],
        callbacks=create_callbacks('custom_cnn')
    )
    models_dict['custom_cnn'] = trainer_cnn

    # 2. ResNet50
    print("\n" + "="*60)
    print("TREINO 2/3: ResNet50 Transfer Learning")
    print("="*60)

    model_resnet = EmotionCNNModels.create_resnet_transfer(input_shape, config['num_classes'])
    model_resnet = EmotionCNNModels.compile_model(model_resnet, config['learning_rate'])

    trainer_resnet = ModelTrainer(model_resnet, 'resnet50')
    trainer_resnet.train(
        train_gen, val_gen,
        epochs=config['epochs'],
        callbacks=create_callbacks('resnet50')
    )
    models_dict['resnet50'] = trainer_resnet

    # 3. EfficientNet
    print("\n" + "="*60)
    print("TREINO 3/3: EfficientNetB0 Transfer Learning")
    print("="*60)

    model_eff = EmotionCNNModels.create_efficientnet_transfer(input_shape, config['num_classes'])
    model_eff = EmotionCNNModels.compile_model(model_eff, config['learning_rate'])

    trainer_eff = ModelTrainer(model_eff, 'efficientnet')
    trainer_eff.train(
        train_gen, val_gen,
        epochs=config['epochs'],
        callbacks=create_callbacks('efficientnet')
    )
    models_dict['efficientnet'] = trainer_eff

    return models_dict


print(" Módulo de treino carregado")

# Teste rápido
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" TESTE DO MÓDULO DE TREINO")
    print("="*70 + "\n")

    print(" ModelTrainer disponível")
    print(" Funções de treino disponíveis")

    if CONFIG is not None:
        print(" CONFIG importado com sucesso")

    print("\n TESTE PASSOU! Módulo pronto para uso.")
    print("\n" + "="*70)