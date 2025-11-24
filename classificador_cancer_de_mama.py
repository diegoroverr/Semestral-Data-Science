import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ==========================================

# Ler arquivo (certifique-se que o breast-cancer.csv está na mesma pasta)
dados = pd.read_csv("breast-cancer.csv")

# Tratar valores ausentes: substituir '?' por vazio e remover linhas incompletas
dados = dados.replace("?", pd.NA)
dados = dados.dropna()

# Converter texto para números (Label Encoding)
# Isso é necessário porque o Sklearn não aceita strings como "left", "premeno", etc.
encoders = {} 
for coluna in dados.columns:
    if dados[coluna].dtype == 'object':
        le = LabelEncoder()
        dados[coluna] = le.fit_transform(dados[coluna])
        encoders[coluna] = le

# Separar atributos (X) e classe alvo (y)
# A coluna alvo chama-se "Class" neste dataset
dados_atributos = dados.drop(columns=["Class"])
dados_classes = dados["Class"]


# ==========================================
# 2. TREINAMENTO DO MODELO
# ==========================================

# Dividir em Treino (70%) e Teste (30%)
atributos_train, atributos_test, classes_train, classes_test = train_test_split(
    dados_atributos, dados_classes, test_size=0.3, random_state=42
)

# Criar e treinar a Árvore de Decisão
tree = DecisionTreeClassifier(random_state=42)
breast_tree = tree.fit(atributos_train, classes_train)

print("Treinamento concluído!")
print(f"Classes aprendidas: {breast_tree.classes_} (0=no-recurrence, 1=recurrence)")


# ==========================================
# 3. AVALIAÇÃO (ACURÁCIA E MATRIZ)
# ==========================================

# Realizar previsões nos dados de teste
predicoes = breast_tree.predict(atributos_test)

# Calcular Acurácia
acuracia = metrics.accuracy_score(classes_test, predicoes)
print(f"\nAcurácia global: {acuracia:.2f} ({acuracia*100:.1f}%)")

# Gerar e mostrar a Matriz de Confusão
print("\nGerando Matriz de Confusão...")
matriz = confusion_matrix(classes_test, predicoes)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Não Recorrente", "Recorrente"])

# Plotar o gráfico
disp.plot()
plt.title("Matriz de Confusão - Câncer de Mama")
plt.show()


# ==========================================
# 4. CLASSIFICAR NOVA INSTÂNCIA
# ==========================================

# Dados de um paciente fictício convertidos para os códigos numéricos gerados pelo LabelEncoder
# Exemplo: age='50-59'(3), menopause='ge40'(0), tumor-size='15-19'(2), etc.
nova_instancia = [[3, 0, 2, 0, 0, 1, 1, 0, 0]]

# Fazer a previsão
classe_inferida = breast_tree.predict(nova_instancia)
probabilidade = breast_tree.predict_proba(nova_instancia)

# Traduzir o resultado numérico de volta para texto (opcional, mas útil)
resultado_texto = encoders['Class'].inverse_transform(classe_inferida)

print("\n--- Resultado para Nova Instância ---")
print(f"Código Predito: {classe_inferida[0]}")
print(f"Diagnóstico: {resultado_texto[0]}")
print(f"Probabilidade: {probabilidade}")