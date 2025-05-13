import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset (replace 'dropout_data.csv' with your file path)
data = pd.read_csv('dropout_data.csv')

# Replace '#N/A' with NaN for proper handling
data.replace('#N/A', np.nan, inplace=True)

# Define features and target
features = [
    'Attendance % in 2023-24', 'Learning Score % in 2023-24', 'Class 2023-24',
    'gender', 'religion', 'caste', 'is_below_poverty_line', 
    'is_antyodaya_anna_yojana', 'is_disadvantaged_group', 'free_transport'
]
target = 'is_dropout'

# Encode target: 'Dropout' -> 1, 'Promoted'/'Repeating' -> 0
data['is_dropout'] = data['is_dropout'].apply(lambda x: 1 if x == 'Dropout' else 0)

# Split features (X) and target (y)
X = data[features]
y = data[target]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define numerical and categorical features
numerical_features = ['Attendance % in 2023-24', 'Learning Score % in 2023-24', 'Class 2023-24']
categorical_features = [
    'gender', 'religion', 'caste', 'is_below_poverty_line', 
    'is_antyodaya_anna_yojana', '[b]is_disadvantaged_group', 'free_transport'
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

# Create Random Forest pipeline with SMOTE for class imbalance
rf_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    ))
])

# Hyperparameter tuning with RandomizedSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 15, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}
random_search = RandomizedSearchCV(
    rf_pipeline, 
    param_distributions=param_grid, 
    n_iter=10, 
    cv=5, 
    scoring='f1', 
    random_state=42, 
    n_jobs=-1
)
random_search.fit(X_train, y_train)

# Best model
best_rf = random_search.best_estimator_
print("Best Hyperparameters:", random_search.best_params_)

# Evaluate on test set
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Dropout', 'Dropout']))

# ROC-AUC and Precision-Recall AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = np.trapz(recall, precision)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Cross-validated Recall
cv_recall = cross_val_score(best_rf, X, y, cv=5, scoring='recall', n_jobs=-1)
print(f"Cross-Validated Recall: {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")

# Feature importance
feature_names = (
    numerical_features + 
    best_rf.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['encode']
    .get_feature_names_out(categorical_features).tolist()
)
importances = best_rf.named_steps['classifier'].feature_importances_
feature_importance = pd.DataFrame(
    {'Feature': feature_names, 'Importance': importances}
).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance.head(10))

# Save the model for deployment
joblib.dump(best_rf, 'rf_dropout_model.pkl')

# Example: Predict for a new student
new_student = pd.DataFrame({
    'Attendance % in 2023-24': [45.0],
    'Learning Score % in 2023-24': [50.0],
    'Class 2023-24': [6],
    'gender': ['Male'],
    'religion': ['Hinduism'],
    'caste': ['OBC'],
    'is_below_poverty_line': ['YES'],
    'is_antyodaya_anna_yojana': ['NO'],
    'is_disadvantaged_group': ['YES'],
    'free_transport': ['NO']
})
prediction = best_rf.predict(new_student)
probability = best_rf.predict_proba(new_student)[:, 1]
print(f"\nNew Student Dropout Prediction: {'Dropout' if prediction[0] == 1 else 'Not Dropout'}")
print(f"Dropout Probability: {probability[0]:.4f}")