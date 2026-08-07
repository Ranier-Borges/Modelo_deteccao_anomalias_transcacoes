# Motor de Detecção de Fraudes em Transaçõe


# 🔎 Detecção de Anomalias em Transações Financeiras

Projeto de **Data Science e Machine Learning para detecção de transações potencialmente fraudulentas**, desenvolvido a partir de técnicas de análise exploratória, tratamento de dados desbalanceados, engenharia de atributos e modelos supervisionados de classificação.

O objetivo principal é identificar padrões associados a transações fraudulentas e construir um modelo capaz de atribuir um **risco de fraude** a cada transação.

---

## 🎯 Objetivo do Projeto

Desenvolver e avaliar modelos de Machine Learning capazes de identificar transações potencialmente fraudulentas em um conjunto de dados financeiros altamente desbalanceado.

O projeto busca responder à seguinte pergunta:

> **Dada uma nova transação financeira, qual é a probabilidade de ela estar associada a uma atividade fraudulenta?**

Para isso, são avaliadas diferentes abordagens de classificação e métricas específicas para problemas de fraude.

---

## 🧩 Contexto do Problema

A detecção de fraude é um problema particularmente desafiador devido ao forte desbalanceamento entre transações legítimas e fraudulentas.

No conjunto de dados utilizado, aproximadamente:

* **99,83%** das transações são legítimas;
* **0,17%** das transações são fraudulentas.

Esse cenário faz com que métricas tradicionais, como **Accuracy**, possam fornecer uma percepção equivocada sobre a qualidade do modelo.

Por exemplo, um modelo que classificasse praticamente todas as transações como legítimas poderia apresentar uma Accuracy extremamente alta e, ao mesmo tempo, ser incapaz de detectar fraudes.

Por esse motivo, o projeto dá maior importância a métricas como:

* Precision;
* Recall;
* F1-Score;
* ROC-AUC;
* Precision-Recall;
* Matriz de Confusão.

---

## 🏗️ Pipeline do Projeto

O fluxo atual do projeto pode ser representado da seguinte forma:

```text
                    Dataset
                       │
                       ▼
              Análise Exploratória
                       │
                       ▼
             Tratamento dos Dados
                       │
                       ▼
              Feature Engineering
                       │
                       ▼
             Train / Test Split
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
     Regressão      Random        XGBoost
     Logística      Forest
          │            │             │
          └────────────┼─────────────┘
                       ▼
              Avaliação dos Modelos
                       │
                       ▼
              Ajuste de Threshold
                       │
                       ▼
             Interpretabilidade
                    (SHAP)
                       │
                       ▼
             Detecção de Fraude
```

---

# 📊 Dataset

O projeto utiliza o conhecido dataset de transações de cartão de crédito disponibilizado para estudos de detecção de fraude.

As principais características do dataset são:

| Característica | Descrição                                  |
| -------------- | ------------------------------------------ |
| `Time`         | Tempo decorrido desde a primeira transação |
| `V1` a `V28`   | Variáveis transformadas por PCA            |
| `Amount`       | Valor da transação                         |
| `Class`        | Variável alvo                              |
| `Class = 0`    | Transação legítima                         |
| `Class = 1`    | Transação fraudulenta                      |

As variáveis `V1` a `V28` são componentes resultantes de uma transformação PCA, realizada para preservar a privacidade das informações originais das transações.

---

# 🔬 Análise Exploratória

A primeira etapa do projeto consiste na análise exploratória dos dados.

Entre os pontos avaliados estão:

* dimensões do dataset;
* tipos das variáveis;
* valores ausentes;
* distribuição das classes;
* distribuição dos valores das transações;
* estatísticas descritivas;
* identificação do desbalanceamento;
* correlação entre variáveis;
* comportamento da variável `Amount`.

Um dos principais achados é o forte desbalanceamento entre as classes.

```text
Transações legítimas
██████████████████████████████████████████████████ 99,83%

Transações fraudulentas
█ 0,17%
```

Esse comportamento influencia diretamente a estratégia de modelagem e avaliação.

---

# ⚙️ Feature Engineering

Foram criadas novas representações para a variável `Amount`, buscando melhorar a capacidade dos modelos de identificar padrões relacionados ao valor das transações.

### Transformação logarítmica

Foi criada a variável:

```python
Amount_log = log1p(Amount)
```

Essa transformação reduz o impacto de valores muito elevados e pode ajudar modelos a trabalhar melhor com uma distribuição altamente assimétrica.

### Normalização

Também foi criada uma versão normalizada do valor da transação:

```python
Amount_scaled
```

A normalização permite colocar essa variável em uma escala mais adequada para determinados algoritmos.

---

# 🤖 Modelos Utilizados

Foram avaliados diferentes algoritmos de classificação.

## 1. Regressão Logística

A Regressão Logística foi utilizada como **baseline**.

Sua principal vantagem é oferecer um modelo simples, rápido e relativamente interpretável para comparação com algoritmos mais complexos.

---

## 2. Random Forest

O Random Forest foi utilizado para capturar relações não lineares entre as variáveis.

Foi considerada a natureza desbalanceada do dataset utilizando:

```python
class_weight="balanced"
```

O algoritmo também permite analisar a importância das variáveis utilizadas na classificação.

---

## 3. XGBoost

O XGBoost foi utilizado como principal modelo de Gradient Boosting.

Esse algoritmo apresenta bom desempenho em problemas tabulares e é especialmente interessante para cenários em que existem relações não lineares entre as variáveis.

Na versão atual do projeto, o XGBoost apresentou o melhor equilíbrio entre Precision, Recall e F1-Score para a classe de fraude.

---

# 📈 Resultados

Os resultados observados na versão atual do projeto são aproximadamente:

| Modelo              | Precision |   Recall | F1-Score |
| ------------------- | --------: | -------: | -------: |
| Regressão Logística |      0,85 |     0,64 |     0,73 |
| Random Forest       |      0,84 |     0,76 |     0,79 |
| **XGBoost**         |  **0,94** | **0,78** | **0,85** |

Os valores demonstram que o **XGBoost apresentou o melhor desempenho geral entre os modelos avaliados**.

Para a classe de fraude, o modelo alcançou aproximadamente:

```text
Precision: 0,94
Recall:    0,78
F1-Score:  0,85
```

### Interpretação

Um Recall de aproximadamente 0,78 indica que o modelo conseguiu identificar cerca de **78% das transações fraudulentas presentes no conjunto avaliado**.

Já uma Precision de aproximadamente 0,94 indica que, entre as transações classificadas como fraude, uma parcela elevada realmente pertencia à classe fraudulenta.

---

# 🎚️ Ajuste do Threshold

Em problemas de fraude, utilizar automaticamente o threshold padrão de `0.5` nem sempre é a melhor estratégia.

Por esse motivo, o projeto também explora a alteração do threshold de classificação.

Exemplo:

```python
threshold = 0.3

y_pred = (y_probability > threshold).astype(int)
```

A alteração do threshold permite controlar o equilíbrio entre:

```text
Recall
   ↕
Detecção de fraudes

Precision
   ↕
Redução de falsos positivos
```

Em um ambiente real, o threshold deve ser definido considerando também o impacto financeiro e operacional de:

* fraude não detectada;
* bloqueio indevido;
* revisão manual;
* perda de receita;
* impacto na experiência do cliente.

---

# 🧠 Interpretabilidade

Além da performance preditiva, o projeto utiliza técnicas de **Explainable AI (XAI)**.

Foi utilizada a biblioteca:

```text
SHAP
```

O objetivo é compreender quais variáveis possuem maior influência nas decisões do modelo.

Isso é especialmente importante em aplicações financeiras, pois não basta saber que uma transação foi classificada como suspeita.

Também é importante compreender:

> **Por que o modelo considerou essa transação suspeita?**

A interpretabilidade pode auxiliar analistas de fraude na investigação das transações classificadas como de alto risco.

---

# 🛡️ Considerações sobre Detecção de Fraude

Uma solução antifraude real não deve depender exclusivamente de um modelo de Machine Learning.

Uma arquitetura mais robusta poderia combinar:

```text
              Transação
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
     Regras    Histórico     ML
     negócio   comportamental
       │          │          │
       └──────────┼──────────┘
                  ▼
             Fraud Score
                  │
          ┌───────┼────────┐
          ▼       ▼        ▼
       Aprovar  Revisar  Bloquear
```

Essa abordagem permite combinar:

* regras de negócio;
* comportamento histórico;
* características da transação;
* Machine Learning;
* score de risco;
* análise humana.

---

# ⚠️ Limitações da Versão Atual

Apesar dos resultados positivos, esta versão deve ser considerada um **protótipo de Data Science / Machine Learning**, e não um motor antifraude pronto para produção.

Entre as principais limitações estão:

### 1. Dataset anonimizado

As variáveis `V1` a `V28` são componentes PCA e não representam diretamente atributos de negócio.

Em um sistema real seria interessante trabalhar com informações como:

* cliente;
* dispositivo;
* localização;
* estabelecimento;
* canal;
* horário;
* frequência de transações;
* comportamento histórico;
* distância entre transações;
* valor médio histórico;
* quantidade de transações em determinado período.

### 2. Validação temporal

A validação atual utiliza divisão entre treino e teste.

Em um cenário real de fraude, uma abordagem temporal seria mais adequada, pois o modelo será utilizado para prever eventos futuros.

### 3. Engenharia de atributos comportamentais

A versão atual ainda não explora profundamente características relacionadas ao comportamento histórico do cliente.

### 4. Threshold

O threshold atual é experimental e deve ser calibrado de acordo com métricas e custos de negócio.

### 5. Produção

O projeto ainda não possui uma arquitetura completa de produção, como:

* API;
* monitoramento;
* versionamento do modelo;
* logging;
* testes automatizados;
* pipeline de treinamento;
* monitoramento de drift;
* controle de versões de dados.

---

# 🚀 Roadmap — Versão 2.0

As próximas evoluções planejadas para o projeto incluem:

## 🔹 Machine Learning

* [ ] Otimização de hiperparâmetros do XGBoost;
* [ ] Avaliação com Cross Validation;
* [ ] Avaliação utilizando PR-AUC;
* [ ] Comparação com LightGBM;
* [ ] Comparação com CatBoost;
* [ ] Otimização de `scale_pos_weight`;
* [ ] Calibração das probabilidades.

## 🔹 Feature Engineering

* [ ] Features temporais;
* [ ] Velocidade de transações;
* [ ] Quantidade de transações por período;
* [ ] Média histórica de valores;
* [ ] Desvio do valor atual em relação ao comportamento histórico;
* [ ] Análise de dispositivo;
* [ ] Análise geográfica;
* [ ] Comportamento por estabelecimento;
* [ ] Features relacionadas ao comportamento do cliente.

## 🔹 Avaliação

* [ ] PR-AUC;
* [ ] Curva Precision-Recall;
* [ ] FPR — False Positive Rate;
* [ ] FNR — False Negative Rate;
* [ ] Análise de custo de fraude;
* [ ] Otimização do threshold;
* [ ] Validação temporal.

## 🔹 Explainable AI

* [ ] SHAP global;
* [ ] SHAP individual;
* [ ] Explicação das principais razões de risco;
* [ ] Geração de justificativa para cada decisão.

## 🔹 Productionização

* [ ] Separação entre notebook e código de produção;
* [ ] API utilizando FastAPI;
* [ ] Docker;
* [ ] Testes automatizados;
* [ ] Logging;
* [ ] Versionamento do modelo;
* [ ] Monitoramento de performance;
* [ ] Monitoramento de data drift;
* [ ] Pipeline de retreinamento.

---

# 🏦 Arquitetura Futura

A evolução proposta para o projeto é transformar o modelo atual em um **Fraud Detection Engine**.

Uma arquitetura futura poderia ser:

```text
                        ┌──────────────────┐
                        │    Transação     │
                        └────────┬─────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
          Regras de        Features de      Histórico do
           Negócio         Transação         Cliente
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │  ML Model     │
                         │   XGBoost     │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │  Fraud Score  │
                         └───────┬───────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
              APROVAR          REVISAR         BLOQUEAR
```

---

# 🧪 Tecnologias

As principais tecnologias utilizadas são:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn
* SHAP
* Jupyter Notebook

---

# 📁 Estrutura Atual

```text
Modelo_deteccao_anomalias_transcacoes/
│
├── deteccao_anomalias_transcacoes.ipynb
└── README.md
```

---

# ▶️ Como Executar

## 1. Clone o repositório

```bash
git clone https://github.com/Ranier-Borges/Modelo_deteccao_anomalias_transcacoes.git
```

## 2. Entre no diretório

```bash
cd Modelo_deteccao_anomalias_transcacoes
```

## 3. Instale as dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap jupyter
```

## 4. Execute o Jupyter Notebook

```bash
jupyter notebook
```

Depois, abra:

```text
deteccao_anomalias_transcacoes.ipynb
```

---

# 📌 Principais Aprendizados

Este projeto demonstra conceitos importantes de Data Science aplicados a um problema realista de detecção de fraude:

* análise exploratória de dados;
* classificação supervisionada;
* tratamento de datasets desbalanceados;
* engenharia de atributos;
* comparação de modelos;
* ajuste de threshold;
* avaliação por Precision, Recall e F1;
* utilização de XGBoost;
* interpretabilidade com SHAP;
* análise de riscos de falsos positivos e falsos negativos.

---

# 🎓 Conclusão

A versão atual demonstra que modelos de Machine Learning podem ser utilizados para identificar padrões associados a transações fraudulentas.

Entre os modelos avaliados, o **XGBoost apresentou o melhor desempenho**, alcançando aproximadamente **94% de Precision, 78% de Recall e 85% de F1-Score para a classe de fraude**.

Entretanto, uma solução antifraude de produção precisa ir além da classificação individual da transação.

O próximo passo natural deste projeto é incorporar **comportamento histórico, engenharia de atributos, validação temporal, otimização de threshold, explicabilidade e uma arquitetura de inferência em tempo real**.

Dessa forma, o projeto poderá evoluir de um modelo experimental de Machine Learning para um verdadeiro:

> ## 🛡️ Fraud Detection Engine

capaz de combinar **Machine Learning + regras de negócio + comportamento transacional + análise de risco**.

---

## 👨‍💻 Autor

**Ranier Borges**

Projeto desenvolvido para estudos e experimentação de técnicas de **Data Science, Machine Learning, análise de risco e detecção de fraudes em transações financeiras**.

