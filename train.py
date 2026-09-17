import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Carregar os dados
# Certifique-se de que o arquivo Excel está na mesma pasta
df = pd.read_excel("seu banco de dados.xlsx")

# 2. Renomear as colunas para facilitar a manipulação
df = df.rename(columns={
    'OT (%)': 'OT', 'DEN (g/cm3)': 'DEN', 'CBR (%)': 'CBR', 
    'LL (%)': 'LL', 'IP (%)': 'IP', 
    2: 'P2_0', 0.42: 'P0_42', 0.074: 'P0_074',
    'Class ': 'Class', 'Class': 'Class_code',
    'σ3': 'sigma3', 'σd': 'sigmad'
})
df['Class'] = df['Class'].fillna('Unknown').astype(str).str.strip()

# 3. Definir features e target de acordo com a pesquisa
# A deformação resiliente (Er) é mantida apenas como auditoria e não entra nas features
features = ['OT', 'DEN', 'CBR', 'LL', 'IP', 'P2_0', 'P0_42', 'P0_074', 'Class', 'sigma3', 'sigmad']
X = df[features]

# O alvo é modelado como ln(1 + MR) devido à alta amplitude
y = np.log1p(df['MR']) 

# 4. Criar o pipeline de pré-processamento e modelagem
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Class'])
    ],
    remainder='passthrough'
)

# Hiperparâmetros validados por GroupKFold na pesquisa
xgb_model = xgb.XGBRegressor(
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    n_estimators=200,
    subsample=1.0,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# 5. Treinar o modelo com a base integral para produção
pipeline.fit(X, y)

# 6. Salvar o modelo treinado para consumo imediato no Streamlit
joblib.dump(pipeline, 'xgb_mr_model.pkl')
print("Modelo treinado e salvo com sucesso como 'xgb_mr_model.pkl'")
