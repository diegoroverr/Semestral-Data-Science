# Abrir os dados
import pandas as pd

# Nomes das colunas do dataset de Cleveland
colunas = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

# Ler arquivo (substituir pelo caminho correto se necessário)
dados = pd.read_csv("processed.cleveland.data", names=colunas)

# Tratar valores ausentes representados por '?'
dados = dados.replace("?", pd.NA)
dados = dados.dropna()

# Converter colunas para float
dados = dados.astype(float)

# Converter target: 0 = sem doença, 1,2,3,4 = com doença
dados["target"] = dados["target"].apply(lambda x: 1 if x > 0 else 0)

# Separar atributos e classes
dados_atributos = dados.drop(columns=["target"])
dados_classes = dados["target"]


# ---- 4. Treinar o modelo (igual ao exemplo) ----
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, classes_train, classes_test = train_test_split(
    dados_atributos, dados_classes, test_size=0.3
)

# Importar indutor
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

# Treinar o modelo
heart_tree = tree.fit(atributos_train, classes_train)

# Ver classes aprendidas pelo modelo
print("Classes aprendidas:", heart_tree.classes_)


# ---- Pretestar o modelo ----
predicoes = heart_tree.predict(atributos_test)

print("\nComparação real vs predito:")
for i in range(len(classes_test)):
    print(classes_test.iloc[i], " - ", predicoes[i])


# ---- Acurácia ----
from sklearn import metrics
print("\nAcurácia global:", metrics.accuracy_score(classes_test, predicoes))


# ---- Matriz de contingência (versão atualizada) ----
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matriz = confusion_matrix(classes_test, predicoes)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
disp.plot()
plt.show()

print("\nMatriz de Confusão:")
print(matriz)


# ---- Classificar nova instância ----
# Exemplo de nova instância (tem que ter as 13 colunas)
nova_instancia = [[
    52, 1, 0, 138, 200, 0, 1, 170, 0, 1.2, 2, 0, 3
]]

classe_inferida = heart_tree.predict(nova_instancia)
dist_prob = heart_tree.predict_proba(nova_instancia)

print("\nClassificação da nova instância:", classe_inferida)
print("Distribuição probabilística:", dist_prob)
print("Classes:", heart_tree.classes_)
