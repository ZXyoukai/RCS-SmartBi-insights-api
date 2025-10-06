import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Supondo que a coluna alvo é a última
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    # Tratamento de valores nulos
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Normalização/Standardização
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    return X_scaled, y
