import numpy as np

# Carregue as previsões do modelo 1
preds_model1 = np.loadtxt("preds_model1.csv", delimiter=",", skiprows=1)

# Carregue as previsões do modelo 2
preds_model2 = np.loadtxt("preds_model2.csv", delimiter=",", skiprows=1)

# As colunas dos arquivos CSV devem ser as probabilidades de cada classe
# As linhas dos arquivos CSV devem ser as previsões de cada exemplo

# Calcule a média das previsões dos dois modelos
ensemble_preds = (preds_model1 + preds_model2) / 2.0

# Obtenha as classes previstas como o índice do valor máximo de probabilidade
ensemble_classes = np.argmax(ensemble_preds, axis=1)

# Carregue as classes reais dos dados de teste em um tensor do NumPy
true_classes = np.loadtxt("true_classes.csv", delimiter=",", skiprows=1)

# Calcule as métricas de avaliação
accuracy = accuracy_score(true_classes, ensemble_classes)
precision = precision_score(true_classes, ensemble_classes, average="macro")
recall = recall_score(true_classes, ensemble_classes, average="macro")
f1 = f1_score(true_classes, ensemble_classes, average="macro")

loss_avg = AVGMetrics()


# Imprima as métricas de avaliação
print(f"Acurácia: {accuracy:.3f}")
print(f"Precisão: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")