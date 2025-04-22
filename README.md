Case Conversão de Oferta iFood
==================================================

Este projeto tem como objetivo avaliar a efetividade de campanhas de marketing direcionadas a clientes do iFood. 
A proposta envolve o desenvolvimento de uma solução completa, abrangendo desde o pré-processamento dos dados, 
análise descritiva, criação e seleção de features, separação dos conjuntos de treino e teste, validação cruzada e tuning de hiperparâmetros, até a construção dos artefatos necessários para um eventual *deploy* em produção.

O principal objetivo consiste em construir um modelo preditivo capaz de classificar se um cliente, 
após ser impactado por uma campanha de marketing, aceitará ou não a oferta recebida.

# Etapas de desenvolvimento
-------------------------

Para garantir o bom funcionamento do ambiente de desenvolvimento, recomenda-se instalar os pacotes necessários com:

```bash
    pip install -r requirements.txt
```

## 1. Análise exploratória de dados
--------------------------------
A análise exploratória está documentada em [01-eda_raw.ipynb](notebooks/01-eda_raw.ipynb) e [01-eda_interim.ipynb](notebooks/01-eda_interim.ipynb).
Foram realizadas análises de volumetria, univariadas e bivariadas, além da investigação de possíveis casos de data leakage.
Esta etapa orientou a construção da base processada.

## 2. Geração da base processada
-----------------------------
Transforma a base original em uma versão consolidada, pronta para a modelagem.

```bash
    python src/data/pre_processing.py
```

O arquivo resultante será salvo na pasta data/processed.

## 3. Geração da base interim
--------------------------
Criação de novas features com apoio das classes em src/utils/transformers.py.

```bash
    python src/features/build_features.py
``` 

O resultado será salvo em data/interim.

## 4. Split das bases de treino e teste
------------------------------------
Separação dos conjuntos de treino e teste de forma estratificada, garantindo representatividade da variável alvo.

```bash
    python src/data/train_test_split.py
```

Os conjuntos serão salvos em data/train_test.

## 5. Feature Selection
--------------------
Técnica Boruta, baseada em florestas aleatórias, utilizada para selecionar as variáveis mais relevantes.

```bash
    python src/features/feature_selection.py
```

Resultado salvo em ([features_selected.yaml](src/features/selected/features_selected.yaml))

## 6. Geração dos Encoders
-----------------------
Geração dos encoders e bases codificadas, incluindo preenchimento de nulos, padronização de strings e tratamento de variáveis categóricas.

```bash
    python src/features/create_encoders.py
```

## 7. Model selection
------------------
Modelos avaliados: Decision Tree, Random Forest, GBT, AdaBoost, XGBoost e LightGBM.
Validação feita com cross-validation estratificada. 

## 8. Tuning de hiperparâmetros
----------------------------
Modelo com melhor desempenho ajustado com Optuna, utilizando otimização bayesiana.

```bash
    python src/models/tuning.py
```

## 9. Treinamento final do modelo
------------------------------
Treinamento final documentado em notebooks/05-Model.ipynb.
O modelo será salvo em models/predictors.

## 10. Geração dos artefatos para produção
---------------------------------------
Criação da pipeline final com todos os componentes necessários para execução em produção.
O modelo recebe um JSON como entrada e retorna um score.

```bash
    python src/models/generate_artifacts.py
```

------------

Organização do projeto


    ├── README.md                       <- README do projeto para guiar a sua execução.
    │
    ├── data
    │   ├── interim                     <- Dados intermediários, transformados a partir dos dados raw.
    │   ├── processed                   <- Dados finais: dados canônicos para o processo de modelagem.
    │   └── raw                         <- Dados raw, originais e imutáveis.
    │
    ├── notebooks                       <- Jupyter notebooks para análise descritiva, 
    │                                      acompanhamento do desenvolvimento e processos interativos.
    │
    ├── requirements.txt                <- Lista de dependências do projeto. 
    │                                      Gerado com `pip freeze > requirements.txt`.
    │
    ├── pyproject.toml                  <- Torna o projeto instalável (`pip install -e .`). 
    │                                      Permite que `src` possa ser importado como módulo.
    │
    ├── src                             <- Código-fonte do projeto.
    │   │
    │   ├── __init__.py                 <- Torna `src` um módulo Python.
    │   │
    │   ├── data                        <- Scripts de processamento intermediário.
    │   │   └── pre_processing.py
    │   │   └── pre_processing.py
    │   │   └── train_test_split.py
    │   │
    │   ├── features                    <- Scripts de criação e seleção de features.
    │   │   └── build_features.py
    │   │   └── create_encoders.py
    │   │   └── feature_selection.py
    │   │   └── selected                <- Contém os arquivos de features selecionadas.
    │   │
    │   ├── models                      <- Scripts relacionados à modelagem.
    │   │   └── generate_artifacts.py
    │   │   └── tunning.py
    │   │
    │   └── utils                       <- Funções auxiliares internas do projeto.
    │
    └── models                          <- Modelos treinados e serializados 
        └── encoders                    <- Encoders utilizados no pipeline.
        └── predictors                  <- Modelo predtivo.

--------


<p><small>Projeto baseado em <a target="_blank" href=https://drivendata.github.io/cookiecutter-data-science/>cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>