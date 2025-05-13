#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score


# In[3]:


data = pd.read_excel(r'D:\dse_data\input.xlsx')
data


# In[4]:


print("Data Overview:")
print(data.head())
print("\nData Summary:")
print(data.info())
print("\nMissing Values per Column:")
print(data.isnull().sum())


# In[5]:


drop_columns = ['district_name', 'block_name', 'cluster_name', 'udise_id', 'school_name', 'school_address', 
                'unique_student_identifier', 'student_name', 'uuid', 'longitude', 'latitude', 'Class in 24-25', 'A', 'B']
data_filtered = data.drop(columns=drop_columns)


# In[6]:


data_filtered['is_dropout'] = data_filtered['is_dropout'].apply(lambda x: 1 if x == 'Dropout' else 0)


# In[7]:


plt.figure(figsize=(6, 4))
sns.countplot(x='is_dropout', data=data_filtered)
plt.title("Distribution of Dropout Status")
plt.show()


# In[10]:



categorical_columns = data_filtered.select_dtypes(include=['object']).columns.tolist()
numerical_columns = data_filtered.select_dtypes(exclude=['object']).columns.tolist()


for col in categorical_columns:
    data_filtered[col] = data_filtered[col].astype('category')


for col in categorical_columns:
    
    if 'missing' not in data_filtered[col].cat.categories:
        data_filtered[col] = data_filtered[col].cat.add_categories('missing')
    
    
    data_filtered[col] = data_filtered[col].fillna('missing')


# In[11]:


X = data_filtered.drop(columns=['is_dropout'])
y = data_filtered['is_dropout']


# In[12]:


data_filtered[categorical_columns] = data_filtered[categorical_columns].astype('category')


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[14]:


catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, random_seed=42, 
                                    cat_features=categorical_columns, verbose=0)
catboost_model.fit(X_train, y_train)


# In[21]:


cv_scores_catboost = cross_val_score(catboost_model, X_train, y_train, cv=5, scoring='accuracy')

print("CatBoost Cross-Validation Accuracy Scores:", cv_scores_catboost)
print("Mean CatBoost CV Accuracy:", cv_scores_catboost.mean())


# In[22]:


y_pred_catboost = catboost_model.predict(X_test)
y_pred_proba_catboost = catboost_model.predict_proba(X_test)[:, 1]


# In[23]:


conf_matrix_catboost = confusion_matrix(y_test, y_pred_catboost)
class_report_catboost = classification_report(y_test, y_pred_catboost)

print("CatBoost Confusion Matrix:\n", conf_matrix_catboost)
print("\nCatBoost Classification Report:\n", class_report_catboost)


# In[24]:


roc_auc_catboost = roc_auc_score(y_test, y_pred_proba_catboost)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)

print(f"CatBoost Test Accuracy: {accuracy_catboost:.2f}")
print(f"CatBoost ROC AUC Score: {roc_auc_catboost:.2f}")


# In[25]:


feature_importances_catboost = catboost_model.get_feature_importance()
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances_catboost, y=X.columns)
plt.title("Feature Importances from CatBoost Model")
plt.show()


# In[ ]:




