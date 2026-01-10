# Reconhecimento de Emocoes atraves de Posturas Corporais

Projeto de Aprendizagem Profunda - Universidade Catolica Portuguesa

**Autor:** Daniel Rodrigues  
**Data:** Janeiro 2026

## Descricao

Sistema de classificacao de emocoes baseado exclusivamente em posturas corporais, removendo deliberadamente informacao facial para focar em sinais nao-verbais do corpo. Utiliza o dataset EMOTIC com 26 categorias emocionais e arquitetura ResNet50 com transfer learning.

## Dataset

- **EMOTIC** (EMOTions In Context)
- 23,788 anotacoes em 18,316 imagens
- 26 categorias emocionais
- Desbalanceamento significativo (Engagement: 43.9%, Excitement: 1.5%)

## Metodologia

### Pre-processamento
- Remocao automatica de rostos (MediaPipe Face Detection + Gaussian Blur)
- Extracao de bounding boxes individuais
- Normalizacao para 224x224 pixels

### Modelo
- **Arquitetura:** ResNet50 com Transfer Learning
- **Parametros:** ~24M total (~2M treinaveis)
- **Configuracao:**
  - Otimizador: Adam (lr=0.001)
  - Epocas: 20
  - Batch size: 32
  - Loss: Categorical Crossentropy

## Estrutura do Projeto

```
.
├── config.py                  # Configuracoes globais
├── data_loader.py            # Carregamento e geradores de dados
├── cnn_models.py             # Definicao da arquitetura ResNet50
├── face_removal.py           # Remocao de rostos (MediaPipe/OpenCV)
├── training.py               # Logica de treino
├── evaluation.py             # Metricas e avaliacao
├── train.py                  # Script principal de treino
├── load_emotic_direct.py     # Carregamento do dataset EMOTIC
├── visualize_emotic.py       # Visualizacao do dataset
└── requirements.txt          # Dependencias
```

## Instalacao

```bash
# Instalar dependencias
pip install -r requirements.txt

# Carregar dataset EMOTIC
python load_emotic_direct.py

# Treinar modelo
python train.py
```

## Requisitos

- Python 3.8+
- TensorFlow 2.18.0
- OpenCV 4.8.0+
- MediaPipe 0.10.0+
- NumPy, Pandas, Scikit-learn

## Resultados

### Metricas Globais
- **Accuracy:** 30.42%
- **F1-Score (Macro):** 0.0179
- **F1-Score (Weighted):** 0.1419

### Analise
O modelo apresentou overfitting extremo para a classe majoritaria (Engagement), classificando praticamente todas as amostras nesta categoria. Das 26 emocoes, apenas Engagement foi aprendida (Recall: 1.0, Precision: 0.3042).

### Causas da Baixa Performance
1. **Desbalanceamento severo** (ratio 30:1) sem tecnicas de mitigacao
2. **Epocas insuficientes** (20) para convergencia adequada
3. **Complexidade da tarefa** (26 categorias apenas por posturas corporais)

## Limitacoes e Trabalho Futuro

### Limitacoes Identificadas
- Apenas 20 epocas de treino
- Sem tecnicas de balanceamento (class weights, SMOTE)
- Ausencia de comparacao com outras arquiteturas
- Sem informacao temporal (apenas imagens estaticas)

### Recomendacoes
1. Aumentar epocas (50-100+) e aplicar class weights
2. Reduzir numero de classes para emocoes basicas (6-8)
3. Comparar multiplas arquiteturas (CNN custom, EfficientNet)
4. Incorporar informacao temporal (video)
5. Usar keypoints de pose (OpenPose/MediaPipe Pose)
6. Testar em outros datasets (BodyTalk, HAPPEI)

## Implicacoes Eticas

- **Privacidade:** Remocao facial reduz identificacao biometrica
- **Consentimento:** Necessario para qualquer aplicacao real
- **Supervisao humana:** Obrigatoria dado baixa accuracy
- **Uso apropriado:** Nao adequado para decisoes criticas

## Ficheiros Gerados

Apos execucao do treino:
- `models/emotion_resnet50_final.h5` - Modelo treinado
- `models/label_encoder.pkl` - Encoder de labels
- `results/confusion_matrix.png` - Matriz de confusao
- `results/classification_report.txt` - Relatorio detalhado
- `results/training_history.png` - Curvas de treino

## Referencia

Para mais detalhes, consultar o relatorio completo:
`Relatorio_Final_Emocoes_Posturais.docx`

## Licenca

Projeto academico - Universidade Catolica Portuguesa
