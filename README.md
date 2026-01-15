import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv("/content/insurance.csv")
print("Dataset Loaded Successfully")
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data['region'] = le.fit_transform(data['region'])
X_reg = data.drop('expenses', axis=1)
y_reg = data['expenses']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
regressor = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)
print("\n--- REGRESSION RESULTS ---")
print("MAE :", mean_absolute_error(y_test_reg, y_pred_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
print("R2 Score:", r2_score(y_test_reg, y_pred_reg))
def risk_label(expenses):
    if expenses < 10000:
        return 0   # Low Risk
    elif expenses < 20000:
        return 1   # Medium Risk
    else:
        return 2   # High Risk
data['risk'] = data['expenses'].apply(risk_label)
X_clf = data.drop(['expenses', 'risk'], axis=1)
y_clf = data['risk']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)
classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
classifier.fit(X_train_clf, y_train_clf)
y_pred_clf = classifier.predict(X_test_clf)
print("\n--- CLASSIFICATION RESULTS ---")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("\nClassification Report:")
print(classification_report(
    y_test_clf, y_pred_clf,
    target_names=['Low Risk', 'Medium Risk', 'High Risk']
))
plt.figure()
sns.heatmap(confusion_matrix(y_test_clf, y_pred_clf),
            annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Risk Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
new_customer = np.array([[40, 1, 29.4, 2, 1, 2]])
predicted_premium = regressor.predict(new_customer)[0]
predicted_risk_class = classifier.predict(new_customer)[0]
risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
print("\n--- NEW CUSTOMER PREDICTION ---")
print("Predicted Insurance Premium:", predicted_premium)
print("Predicted Risk Category:", risk_map[predicted_risk_class])
