"""
This module implements a machine learning pipeline for breast cancer diagnosis 
using the XGBoost classifier.
The pipeline includes data preprocessing, model training, hyperparameter tuning with Optuna
and evaluation  of the model's performance. 
The dataset used for this pipeline is a breast cancer dataset loaded from a CSV file on Kaggle.com.
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

df = pd.read_csv("C:/Users/Marlon/OneDrive/Dokumente/BreastCancerXGBoost/archive/data.csv")
df.head()

# Check empty columns
df.info()
df.columns[df.isnull().any()].tolist()

# Delete empty cloumns
df.drop(['Unnamed: 32'], axis=1, inplace=True)

# Separation of training and test-datasets
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Classification: malignant: 1, beneign: 0
y = y.map({'M':1, 'B':0})

# Define test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objective function for optuna
def xgb_objective(trial):
    """
    Objective function for Optuna hyperparameter optimization of an XGBoost classifier.

    Suggests hyperparameters for the XGBoost model and evaluates its accuracy on the test dataset.

    Suggested Hyperparameters:
        - n_estimators: Number of boosting rounds.
        - max_depth: Maximum tree depth.
        - learning_rate: Step size shrinkage.
        - subsample: Subsample ratio of training instances.
        - colsample_bytree: Subsample ratio of columns.
        - reg_alpha: L1 regularization term.
        - reg_lambda: L2 regularization term.

    Args:
        trial (optuna.trial.Trial): Trial object for suggesting hyperparameters.

    Returns:
        float: Accuracy of the model on the test dataset.
    """
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }

    xgb_model = xgb.XGBClassifier(**param, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Study and optimize optuna output
xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=50)

print("Best trial: (XGBoost)")
xgb_best_trial = xgb_study.best_trial
print(f"  Value: {xgb_best_trial.value}")
print("  Params: ")
for key, value in xgb_best_trial.params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
xgb_best_params = xgb_best_trial.params
xgb_final_model = xgb.XGBClassifier(**xgb_best_params, random_state=42)
xgb_final_model.fit(X_train, y_train)
xgb_final_y_pred = xgb_final_model.predict(X_test)

# Confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, xgb_final_y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Report
print(classification_report(y_test, xgb_final_y_pred))

# Feature importance
xgb.plot_importance(xgb_final_model)
plt.rcParams['figure.figsize'] = [12, 9]
plt.show()
