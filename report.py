import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_reports(results, y_test):
    for name, res in results.items():
        print(f"\n===== Modelo: {name} =====")
        print(f"Acurácia: {res['accuracy']:.4f}")
        print(f"Precisão: {res['precision']:.4f}")
        print(f"Recall: {res['recall']:.4f}")
        print(f"F1-score: {res['f1']:.4f}")
        if res['roc_auc'] is not None:
            print(f"ROC-AUC: {res['roc_auc']:.4f}")
        print("\nRelatório de Classificação:")
        print(res['classification_report'])
        # Matriz de confusão
        plt.figure(figsize=(5,4))
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {name}')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.show()
        # Para multiclasse, não plotamos curva ROC individual
        print(f"Nota: Dataset multiclasse detectado - curvas ROC específicas omitidas.")
    # Geração de insights automáticos
    generate_insights(results)

def generate_insights(results):
    print("\n===== Insights Automáticos =====")
    best_model = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"Melhor modelo (F1-score): {best_model}")
    for name, res in results.items():
        print(f"\nModelo: {name}")
        if res['recall'] < 0.7:
            print("- O modelo está com recall baixo. Pode estar errando ao identificar positivos.")
        if res['precision'] < 0.7:
            print("- O modelo está com precisão baixa. Pode estar gerando muitos falsos positivos.")
        if res['accuracy'] < 0.7:
            print("- A acurácia geral está baixa. Considere revisar os dados ou o modelo.")
        print("Sugestões:")
        print("- Tente ajustar hiperparâmetros (ex: GridSearchCV)")
        print("- Experimente feature engineering (novas variáveis)")
        print("- Avalie coletar mais dados se possível")
        print("- Teste outros algoritmos ou técnicas avançadas")
