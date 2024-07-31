#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:07:54 2024

@author: pavelkim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay

df = pd.read_csv('desktop/survey.csv')

df = df[df['Gender'].isin(['Male', 'Female'])]

print(df.head())

print(df.describe(include='all'))

missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

df.fillna(method='ffill', inplace=True)

kooth_colors = ['#004080', '#66CCFF', '#66FF99', '#CC66FF', '#FF66B2']

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Age', palette=kooth_colors)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

gender_counts = df['Gender'].value_counts()
large_gender_counts = gender_counts[gender_counts > 50]
small_gender_counts = pd.Series(gender_counts[gender_counts <= 50].sum(), index=['Rest'])
gender_counts = pd.concat([large_gender_counts, small_gender_counts])

plt.figure(figsize=(8, 8))
gender_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=kooth_colors)
plt.title('Gender Distribution')
plt.ylabel('')
plt.show()

country_counts = df['Country'].value_counts()
large_country_counts = country_counts[country_counts > 20]
small_country_counts = pd.Series(country_counts[country_counts <= 20].sum(), index=['Rest'])
country_counts = pd.concat([large_country_counts, small_country_counts])

plt.figure(figsize=(10, 10))
country_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=kooth_colors)
plt.title('Country Distribution')
plt.ylabel('')
plt.show()

df_encoded = df.copy()
for column in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[column] = df_encoded[column].astype('category').cat.codes

plt.figure(figsize=(14, 10))
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

treatment_counts = df['treatment'].value_counts()
print("Class imbalance in 'treatment':\n", treatment_counts)

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='treatment', palette=kooth_colors)
plt.title('Treatment Seeking Distribution')
plt.xlabel('Treatment')
plt.ylabel('Count')
plt.grid(True)
plt.show()

X = df.drop(columns=['treatment'])
y = df['treatment'].apply(lambda x: 1 if x == 'Yes' else 0)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('selector', SelectKBest(score_func=f_classif)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'selector__k': [5, 10, 15, 'all'],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print("Best Parameters: ", grid_search.best_params_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Treatment', 'Treatment'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.grid(False)
plt.show()

y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color=kooth_colors[0], lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), cv_scores, marker='o', color=kooth_colors[1])
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
