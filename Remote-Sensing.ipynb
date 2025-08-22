import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import joblib

# Synthetic dataset 
np.random.seed(42)
img_rows, img_cols = 120, 150
num_samples = img_rows * img_cols

spectral_bands = np.random.rand(num_samples, 6)
temperature = 15 + 10 * np.random.rand(num_samples) 
elevation = 50 + 200 * np.random.rand(num_samples) 

# NDVI calculation
ndvi = (spectral_bands[:, 3] - spectral_bands[:, 2]) / (spectral_bands[:, 3] + spectral_bands[:, 2] + 1e-6)
ndvi = ndvi.reshape(-1, 1)

# SAVI (Soil Adjusted Vegetation Index) calculation 
L = 0.5
savi = ((spectral_bands[:, 3] - spectral_bands[:, 2]) * (1 + L)) / (spectral_bands[:, 3] + spectral_bands[:, 2] + L + 1e-6)
savi = savi.reshape(-1,1)

features = np.hstack((spectral_bands, temperature.reshape(-1,1), elevation.reshape(-1,1), ndvi, savi))

labels = []
for i, b in enumerate(spectral_bands):
    if b[3] > 0.6 and ndvi[i] > 0.3:
        labels.append('Greenery')
    elif b[3] < 0.2 and b[0] < 0.3:
        labels.append('Water')
    elif b[2] > 0.6 and b[0] > 0.5:
        labels.append('Built-up')
    else:
        labels.append('Barren')

df = pd.DataFrame(features, columns=[
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Temperature', 'Elevation', 'NDVI', 'SAVI'])
df['LandCover'] = labels

rows = np.repeat(np.arange(img_rows), img_cols)
cols = np.tile(np.arange(img_cols), img_rows)
df['Row'] = rows
df['Col'] = cols

le = LabelEncoder()
df['LandCoverEnc'] = le.fit_transform(df['LandCover'])

X = df.drop(columns=['LandCover', 'LandCoverEnc', 'Row', 'Col'])
y = df['LandCoverEnc']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    results[name] = (model, acc)

best_model = results['Random Forest'][0]

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test[:1000])

shap.summary_plot(shap_values, features=X_test[:1000], feature_names=X.columns)

joblib.dump(best_model, 'landcover_best_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("Best model and scaler saved.")

# 1. Feature distributions across land cover classes
plt.figure(figsize=(12, 8))
for i, feature in enumerate(X.columns[:6], 1):  # Just first 6 spectral bands for clarity
    plt.subplot(2, 3, i)
    sns.kdeplot(data=df, x=feature, hue='LandCover', common_norm=False)
    plt.title(f'Distribution of {feature} by Land Cover')
plt.tight_layout()
plt.show()

# 2. Correlation heatmap of features
plt.figure(figsize=(10,8))
corr = df[X.columns].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

# 3. Pairplot of selected key features with class hue
sns.pairplot(df, vars=['NDVI', 'SAVI', 'Elevation', 'Temperature'], hue='LandCover', corner=True, height=2.5)
plt.suptitle('Pairplot of Key Features by Land Cover', y=1.02)
plt.show()

# 4. Bar plot for sample count distribution across classes
plt.figure(figsize=(7,5))
count_series = df['LandCover'].value_counts()
sns.barplot(x=count_series.index, y=count_series.values, palette='Set2')
plt.title('Sample Count per Land Cover Class')
plt.ylabel('Number of Samples')
plt.xlabel('Land Cover Class')
plt.show()

# 5. Plot model accuracies comparison
plt.figure(figsize=(7,4))
model_names = list(results.keys())
accuracies = [results[m][1] for m in model_names]
sns.barplot(x=model_names, y=accuracies, palette='pastel')
plt.ylim(0,1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.3f}", ha='center')
plt.show()
