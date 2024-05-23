import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Marlon/OneDrive/Dokumente/BreastCancerXGBoost/archive/data.csv")

df.head()
df.info()

df.columns[df.isnull().any()].tolist()
# Löschen der Spalte mit fehlenden Werten
df.drop(['Unnamed: 32'], axis=1, inplace=True)

# Separation der Zielvariable und den Werten
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Kategorisierung der beiden Variablen
# Bösrtige Tumore: 1, Gutartige Tumore: 0
y = y.map({'M':1, 'B':0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Vorhersagen basierend auf den Testdaten
y_pred = model.predict(X_test)

# Genauigkeit des Modells, also den Anteil der korrekt vorhergesagten Ergebnisse im Verhältnis zu allen Vorhersagen
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Konfusionsmatrix mit Heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Report
print(classification_report(y_test, y_pred)) 
# Feature importance: welche Werte haben den größten Einfluss auf das Modell?
xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [12, 9]
plt.show() 