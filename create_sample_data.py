from sklearn.datasets import load_iris
import pandas as pd

# Carregando dataset Iris para exemplo
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Salvando como CSV
df.to_csv('dados.csv', index=False)
print("Dataset de exemplo criado: dados.csv")