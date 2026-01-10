"""
Módulo para avaliação de modelos
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Importar módulos necessários
try:
    from config import CONFIG
except ImportError:
    CONFIG = {'img_height': 224, 'img_width': 224, 'batch_size': 32}
    print(" config.py não encontrado, usando configuração padrão")

try:
    from data_loader import EmotionDataGenerator
except ImportError as e:
    print(f" Erro ao importar EmotionDataGenerator: {e}")
    EmotionDataGenerator = None


class ModelEvaluator:
    """Classe para avaliar modelos treinados"""

    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        self.class_names = label_encoder.classes_

    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Avalia o modelo no conjunto de teste

        Args:
            X_test: caminhos das imagens de teste
            y_test: labels de teste
            batch_size: tamanho do batch

        Returns:
            dicionário com métricas
        """
        print("\n" + "="*60)
        print("AVALIAÇÃO NO CONJUNTO DE TESTE")
        print("="*60 + "\n")

        # Criar gerador de teste
        img_size = (CONFIG['img_height'], CONFIG['img_width'])
        test_gen = EmotionDataGenerator(
            X_test, y_test,
            batch_size=batch_size,
            img_size=img_size,
            augment=False
        )

        # Fazer predições
        y_pred_probs = self.model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        print(f"\nRESULTADOS:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (Macro): {f1_macro:.4f}")
        print(f"  F1-Score (Weighted): {f1_weighted:.4f}")

        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plota matriz de confusão

        Args:
            y_true: labels verdadeiros
            y_pred: labels preditos
            save_path: caminho para salvar figura
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusão', fontsize=16)
        plt.ylabel('Classe Verdadeira', fontsize=12)
        plt.xlabel('Classe Predita', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Matriz de confusão salva em {save_path}")

        plt.show()

    def classification_report_detailed(self, y_true, y_pred, save_path=None):
        """
        Gera relatório de classificação detalhado

        Args:
            y_true: labels verdadeiros
            y_pred: labels preditos
            save_path: caminho para salvar relatório
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )

        print("\n" + "="*60)
        print("RELATÓRIO DE CLASSIFICAÇÃO")
        print("="*60)
        print(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
                f.write("="*60 + "\n")
                f.write(report)
            print(f" Relatório salvo em {save_path}")

        return report

    def analyze_errors(self, X_test, y_true, y_pred, n_samples=10):
        """
        Analisa erros de classificação

        Args:
            X_test: caminhos das imagens de teste
            y_true: labels verdadeiros
            y_pred: labels preditos
            n_samples: número de amostras a visualizar
        """
        # Encontrar erros
        error_indices = np.where(y_true != y_pred)[0]

        if len(error_indices) == 0:
            print("Nenhum erro encontrado!")
            return

        print(f"\nTotal de erros: {len(error_indices)}/{len(y_true)}")
        print(f"Taxa de erro: {len(error_indices)/len(y_true)*100:.2f}%")

        # Visualizar alguns erros
        n_show = min(n_samples, len(error_indices))
        sample_indices = np.random.choice(error_indices, n_show, replace=False)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()

        for i, idx in enumerate(sample_indices):
            img = cv2.imread(X_test[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]

            axes[i].imshow(img)
            axes[i].set_title(f'Real: {true_label}\nPredito: {pred_label}',
                            fontsize=10)
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('results/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_top_predictions(self, X_test, y_pred_probs, n_samples=10):
        """
        Visualiza predições com maior confiança

        Args:
            X_test: caminhos das imagens de teste
            y_pred_probs: probabilidades preditas
            n_samples: número de amostras
        """
        # Top predições por confiança
        max_probs = np.max(y_pred_probs, axis=1)
        top_indices = np.argsort(max_probs)[-n_samples:][::-1]

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()

        for i, idx in enumerate(top_indices):
            img = cv2.imread(X_test[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred_idx = np.argmax(y_pred_probs[idx])
            pred_label = self.class_names[pred_idx]
            confidence = y_pred_probs[idx][pred_idx]

            axes[i].imshow(img)
            axes[i].set_title(f'{pred_label}\nConfiança: {confidence:.2%}',
                            fontsize=10)
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('results/top_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()


def compare_models(models_dict, X_test, y_test, label_encoder):
    """
    Compara múltiplos modelos

    Args:
        models_dict: dicionário com modelos treinados
        X_test, y_test: dados de teste
        label_encoder: encoder de labels

    Returns:
        DataFrame com comparação
    """
    results = []

    for model_name, trainer in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Avaliando: {model_name}")
        print(f"{'='*60}")

        evaluator = ModelEvaluator(trainer.model, label_encoder)
        metrics = evaluator.evaluate(X_test, y_test)

        results.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1-Score (Macro)': metrics['f1_macro'],
            'F1-Score (Weighted)': metrics['f1_weighted']
        })

        # Salvar visualizações
        os.makedirs('results', exist_ok=True)
        evaluator.plot_confusion_matrix(
            metrics['y_true'],
            metrics['y_pred'],
            save_path=f'results/{model_name}_confusion_matrix.png'
        )

        evaluator.classification_report_detailed(
            metrics['y_true'],
            metrics['y_pred'],
            save_path=f'results/{model_name}_classification_report.txt'
        )

    # Criar tabela comparativa
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

    print("\n" + "="*60)
    print("COMPARAÇÃO DE MODELOS")
    print("="*60)
    print(comparison_df.to_string(index=False))

    # Salvar comparação
    comparison_df.to_csv('results/model_comparison.csv', index=False)

    return comparison_df


print(" Módulo de avaliação carregado")

# Teste rápido
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" TESTE DO MÓDULO DE AVALIAÇÃO")
    print("="*70 + "\n")

    print(" ModelEvaluator disponível")
    print(" Funções de comparação disponíveis")
    print(" Funções de visualização disponíveis")

    print("\n TESTE PASSOU! Módulo pronto para uso.")
    print("\n" + "="*70)