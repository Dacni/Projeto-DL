"""
Arquiteturas de redes neurais convolucionais para classificação de emoções
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Importar CONFIG se existir
try:
    from config import CONFIG
except ImportError:
    CONFIG = {'num_classes': 26}
    print(" config.py não encontrado, usando configuração padrão")


class EmotionCNNModels:
    """Classe com diferentes arquiteturas CNN"""

    @staticmethod
    def create_custom_cnn(input_shape, num_classes):
        """
        CNN customizada simples

        Args:
            input_shape: tupla (height, width, channels)
            num_classes: número de classes

        Returns:
            modelo Keras
        """
        model = models.Sequential([
            # Bloco 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Bloco 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Bloco 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Bloco 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classificador
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    @staticmethod
    def create_resnet_transfer(input_shape, num_classes, trainable_layers=10):
        """
        Transfer learning com ResNet50

        Args:
            input_shape: tupla (height, width, channels)
            num_classes: número de classes
            trainable_layers: número de camadas a treinar

        Returns:
            modelo Keras
        """
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        # Congelar camadas base
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

        # Adicionar classificador
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        return model

    @staticmethod
    def create_efficientnet_transfer(input_shape, num_classes):
        """
        Transfer learning com EfficientNetB0

        Args:
            input_shape: tupla (height, width, channels)
            num_classes: número de classes

        Returns:
            modelo Keras
        """
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        return model

    @staticmethod
    def compile_model(model, learning_rate=0.001):
        """
        Compila o modelo com otimizador e métricas

        Args:
            model: modelo Keras
            learning_rate: taxa de aprendizagem
        """
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return model


def create_callbacks(model_name):
    """
    Cria callbacks para treino

    Args:
        model_name: nome do modelo para salvar

    Returns:
        lista de callbacks
    """
    os.makedirs('models', exist_ok=True)

    checkpoint = ModelCheckpoint(
        f'models/{model_name}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    return [checkpoint, early_stop, reduce_lr]


print(" Modelos CNN definidos")

# Teste rápido
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" TESTE DO MÓDULO DE MODELOS CNN")
    print("=" * 70 + "\n")

    print("Testando criação de modelos...\n")

    input_shape = (224, 224, 3)
    num_classes = 26

    # Testar CNN Customizada
    print("1. CNN Customizada...", end=" ")
    try:
        model = EmotionCNNModels.create_custom_cnn(input_shape, num_classes)
        print(f" ({model.count_params():,} parâmetros)")
        del model
    except Exception as e:
        print(f" ERRO: {e}")
        import traceback

        traceback.print_exc()

    # Testar ResNet50
    print("2. ResNet50...", end=" ")
    try:
        model = EmotionCNNModels.create_resnet_transfer(input_shape, num_classes)
        print(f" ({model.count_params():,} parâmetros)")
        del model
    except Exception as e:
        print(f" ERRO: {e}")

    # Testar EfficientNet
    print("3. EfficientNet...", end=" ")
    try:
        model = EmotionCNNModels.create_efficientnet_transfer(input_shape, num_classes)
        print(f" ({model.count_params():,} parâmetros)")
        del model
    except Exception as e:
        print(f" ERRO: {e}")

    # Testar função de callbacks
    print("\n4. Testando callbacks...", end=" ")
    try:
        callbacks = create_callbacks('test_model')
        print(f" ({len(callbacks)} callbacks criados)")
    except Exception as e:
        print(f" ERRO: {e}")

    print("\n TESTE CONCLUÍDO!")
    print("\n" + "=" * 70)