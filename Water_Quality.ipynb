# --- Step 1: Library Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Step 2: Data Loading and Initial Checks ---
df = pd.read_csv('Data.csv')

# Remove index column if present
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Report and drop missing values
print("Null values per column:\n", df.isnull().sum())
df = df.dropna()

# Feature and target selection
features = ['Chloride', 'Organic_Carbon', 'Solids', 'Sulphate', 'Turbidity', 'ph']
target = 'Label'
X, y = df[features], df[target]

# Class balance
print("\nTarget distribution:\n", y.value_counts(normalize=True))

# --- Step 3: Data Splitting and Scaling ---
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=101)
scaler = MinMaxScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)
joblib.dump(scaler, 'minmax_scaler.pkl')
print("\nScaler saved as 'minmax_scaler.pkl'")

# --- Step 4: Model Setup ---
model_dict = {
    'LogReg': LogisticRegression(random_state=101),
    'RF': RandomForestClassifier(random_state=101, n_estimators=120),
    'SVC': SVC(random_state=101, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=7)
}

results = {}

# --- Step 5: Model Training and Metrics ---
for model_name, model in model_dict.items():
    model.fit(X_tr_scaled, y_tr)
    y_pred = model.predict(X_te_scaled)
    cr = classification_report(y_te, y_pred, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)
    results[model_name] = {
        'classification_report': cr,
        'confusion_matrix': cm,
        'model': model
    }
    print(f"{model_name} is trained.")

# --- Step 6: Print Metrics ---
for name, res in results.items():
    print(f"\nModel: {name}")
    print("Precision:", round(res['classification_report']['1']['precision'], 4))
    print("Recall:", round(res['classification_report']['1']['recall'], 4))
    print("F1-Score:", round(res['classification_report']['1']['f1-score'], 4))
    print("Confusion Matrix:\n", res['confusion_matrix'])

# --- Step 7: Draw Confusion Matrices ---
fig, subplots = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Confusion Matrices by Model', fontsize=15)
axes = subplots.ravel()
for idx, (name, res) in enumerate(results.items()):
    sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', ax=axes[idx], cmap='YlGnBu')
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('conf_matrix_comparison.png')
plt.show()

# --- Step 8: Feature Importance for Random Forest ---
plt.figure(figsize=(8, 5))
rf = results['RF']['model']
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
importances.plot(kind='bar', color='orange')
plt.title('Random Forest - Feature Importances')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.show()

# --- Step 9: Save the Best Model (by F1-score) ---
best_model_name = max(results, key=lambda n: results[n]['classification_report']['1']['f1-score'])
joblib.dump(results[best_model_name]['model'], 'top_water_quality_model.pkl')
print(f"Top model ({best_model_name}) saved as 'top_water_quality_model.pkl'")
