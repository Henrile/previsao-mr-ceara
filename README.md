# 🛣️ Previsão do Módulo de Resiliência de Solos do Ceará

Aplicação desenvolvida em **Python** para estimativa do **Módulo de Resiliência (MR)** de materiais empregados em pavimentação, utilizando técnicas de **Aprendizado de Máquina (Machine Learning)**.

O sistema utiliza um modelo de regressão baseado no algoritmo **XGBoost** e disponibiliza uma interface gráfica desenvolvida com **Streamlit**, permitindo que o usuário informe propriedades físicas, granulométricas e parâmetros de estado de tensão do material para obter uma estimativa do Módulo de Resiliência em **MPa**.

---

## 🎯 Objetivo

O objetivo deste projeto é disponibilizar uma ferramenta computacional de apoio à estimativa do **Módulo de Resiliência de solos**, empregando técnicas de Inteligência Artificial como alternativa complementar às análises laboratoriais tradicionais.

A aplicação busca facilitar a realização de estimativas preliminares de MR a partir de características geotécnicas dos materiais e das tensões aplicadas durante o ensaio.

> **Importante:** os resultados fornecidos pelo sistema possuem finalidade de apoio exploratório e científico e não substituem a realização de ensaios laboratoriais normatizados para dimensionamento definitivo de pavimentos.

---

## 🧠 Modelo de Machine Learning

O modelo utilizado para previsão do MR é baseado no algoritmo:

**XGBoost Regressor**

O fluxo geral da aplicação pode ser representado da seguinte forma:

```text
Dados geotécnicos
       │
       ▼
Pré-processamento
       │
       ├── Variáveis numéricas
       │
       └── Classificação AASHTO
               │
               ▼
        One-Hot Encoding
               │
               ▼
          XGBoost
               │
               ▼
       Predição de ln(1 + MR)
               │
               ▼
     Transformação inversa
               │
               ▼
        MR estimado (MPa)
```

Durante o treinamento, o valor de MR é transformado utilizando:

```text
y = ln(1 + MR)
```

Após a previsão, é realizada a transformação inversa:

```text
MR = exp(y) - 1
```

Essa abordagem auxilia no tratamento da elevada amplitude dos valores da variável-alvo.

---

## 📊 Variáveis utilizadas

O modelo considera informações relacionadas às propriedades físicas e granulométricas dos solos, classificação do material e estado de tensões.

| Variável | Descrição                                  |
| -------- | ------------------------------------------ |
| `OT`     | Umidade Ótima (%)                          |
| `DEN`    | Massa Específica Seca Máxima (g/cm³)       |
| `CBR`    | Índice de Suporte Califórnia (%)           |
| `LL`     | Limite de Liquidez (%)                     |
| `IP`     | Índice de Plasticidade (%)                 |
| `P2_0`   | Percentual passante na peneira de 2,0 mm   |
| `P0_42`  | Percentual passante na peneira de 0,42 mm  |
| `P0_074` | Percentual passante na peneira de 0,074 mm |
| `Class`  | Classificação AASHTO                       |
| `sigma3` | Tensão confinante σ₃ (MPa)                 |
| `sigmad` | Tensão desviadora σd (MPa)                 |

A variável-alvo do modelo é:

```text
MR — Módulo de Resiliência (MPa)
```

---

## 🗂️ Estrutura do projeto

```text
previsao-mr-ceara/
│
├── app.py
├── train.py
├── xgb_mr_model.pkl
├── requirements.txt
├── Fluxo de Previsão de Resiliência dos Solos.png
└── README.md
```

### `app.py`

Arquivo principal da aplicação.

Responsável por:

* criação da interface utilizando Streamlit;
* carregamento do modelo treinado;
* entrada dos dados geotécnicos;
* preparação dos dados;
* execução da previsão;
* conversão da previsão para MPa;
* exibição do resultado ao usuário.

### `train.py`

Script responsável pelo treinamento do modelo.

O arquivo realiza:

1. leitura da base de dados;
2. padronização dos nomes das variáveis;
3. seleção das características utilizadas no treinamento;
4. transformação logarítmica do MR;
5. codificação da classificação AASHTO;
6. treinamento do XGBoost;
7. criação do pipeline de Machine Learning;
8. salvamento do modelo utilizando `joblib`.

### `xgb_mr_model.pkl`

Modelo treinado e serializado utilizado diretamente pela aplicação Streamlit.

### `requirements.txt`

Contém as bibliotecas Python necessárias para execução do projeto.

### `Fluxo de Previsão de Resiliência dos Solos.png`

Fluxograma ilustrativo utilizado na interface da aplicação.

---

## ⚙️ Tecnologias utilizadas

O projeto utiliza principalmente:

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **XGBoost**
* **Scikit-learn**
* **Joblib**
* **OpenPyXL**

---

## 🚀 Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/Henrile/previsao-mr-ceara.git
```

Entre na pasta:

```bash
cd previsao-mr-ceara
```

---

## 2. Criar um ambiente virtual

É recomendável utilizar um ambiente virtual Python.

### Windows

```bash
python -m venv venv
```

Ative o ambiente:

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
```

Ative:

```bash
source venv/bin/activate
```

---

## 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

As principais dependências são:

```text
streamlit
pandas
numpy
xgboost
scikit-learn
joblib
openpyxl
```

---

## 4. Executar a aplicação

Com o modelo `xgb_mr_model.pkl` disponível na pasta do projeto, execute:

```bash
streamlit run app.py
```

O Streamlit disponibilizará um endereço local, normalmente:

```text
http://localhost:8501
```

Abra esse endereço no navegador.

---

## 🖥️ Utilização da aplicação

Na interface, informe os valores correspondentes às propriedades do material:

```text
Umidade Ótima
Massa Específica Seca Máxima
CBR
Limite de Liquidez
Índice de Plasticidade
Passante em 2,0 mm
Passante em 0,42 mm
Passante em 0,074 mm
Classificação AASHTO
Tensão confinante
Tensão desviadora
```

Em seguida, clique em:

```text
Calcular Módulo de Resiliência
```

O sistema processará as informações e apresentará:

```text
Módulo de Resiliência (MR) Previsto
```

em **MPa**.

---

## 🏗️ Treinamento do modelo

Caso seja necessário treinar novamente o modelo, disponibilize a base de dados em formato Excel e ajuste, em `train.py`, o endereço:

```python
df = pd.read_excel("seu banco de dados.xlsx")
```

Depois execute:

```bash
python train.py
```

Ao final do processamento será criado:

```text
xgb_mr_model.pkl
```

O arquivo será utilizado automaticamente pelo `app.py`.

---

## 🔬 Pipeline de processamento

O projeto utiliza um pipeline do **Scikit-learn** que combina o pré-processamento com o modelo XGBoost.

A variável categórica:

```text
Classificação AASHTO
```

é processada utilizando:

```text
OneHotEncoder
```

com tratamento para categorias não observadas anteriormente.

As demais variáveis são enviadas diretamente ao modelo.

O pipeline completo é então serializado utilizando:

```python
joblib.dump()
```

permitindo utilizar exatamente o mesmo processo de transformação durante a inferência.

---

## 🌱 Aplicações

A ferramenta pode auxiliar em estudos relacionados a:

* pavimentação rodoviária;
* mecânica dos pavimentos;
* caracterização geotécnica de solos;
* dimensionamento mecanístico-empírico;
* estudos de materiais para infraestrutura viária;
* modelagem de propriedades mecânicas;
* Inteligência Artificial aplicada à Engenharia Civil;
* previsão de propriedades de solos;
* estudos relacionados ao Módulo de Resiliência.

---

## ⚠️ Limitações

O modelo apresenta estimativas baseadas nos dados utilizados durante seu treinamento.

Portanto, a precisão da previsão depende de fatores como:

* representatividade da base de dados;
* qualidade das informações fornecidas;
* faixa de valores presente durante o treinamento;
* classificação geotécnica dos materiais;
* condições de ensaio;
* domínio de aplicação do modelo.

Predições realizadas para materiais muito diferentes daqueles presentes na base de treinamento devem ser analisadas com cautela.

---

## 📌 Observação científica

O **Módulo de Resiliência (MR)** é um importante parâmetro utilizado na caracterização do comportamento mecânico de materiais empregados em estruturas de pavimentos.

A utilização de técnicas de Machine Learning possibilita investigar relações não lineares entre propriedades físicas, granulométricas, classificatórias e estados de tensão, permitindo o desenvolvimento de modelos auxiliares para previsão dessa propriedade.

Neste projeto, o uso do XGBoost busca explorar essas relações e disponibilizar uma ferramenta computacional para realização de estimativas de maneira rápida e acessível.

---

## 👨‍💻 Autor

**Pedro Henrile**

GitHub:

```text
Henrile
```

Repositório:

```text
https://github.com/Henrile/previsao-mr-ceara
```

---

## 📚 Contexto de pesquisa

Este repositório está relacionado ao desenvolvimento e aplicação de métodos de **Inteligência Artificial para previsão do Módulo de Resiliência de solos**, com ênfase na utilização de características geotécnicas e estados de tensão como variáveis preditoras.

O trabalho integra conceitos de:

**Engenharia Geotécnica + Pavimentação + Ciência de Dados + Machine Learning**

---

## 📄 Licença

Este projeto possui finalidade acadêmica e científica.

Caso o código, modelo ou resultados sejam utilizados em trabalhos acadêmicos, publicações ou projetos derivados, recomenda-se citar adequadamente o trabalho e o respectivo repositório.

---

## ⭐ Contribuições

Sugestões de melhorias, correções e contribuições são bem-vindas.

Você pode utilizar as ferramentas do GitHub para:

* abrir uma **Issue**;
* sugerir melhorias;
* enviar um **Pull Request**;
* relatar problemas;
* propor novos modelos ou variáveis.

---

### 🛣️ Previsão de MR com Inteligência Artificial

> Aplicação de Machine Learning para estimativa do Módulo de Resiliência de solos a partir de características geotécnicas e condições de tensão.
