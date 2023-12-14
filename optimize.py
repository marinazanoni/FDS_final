
# Ignore all warnings
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn import metrics
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

def run(model, param_grid, X, y):
  k = 5
  kf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

  # Dataframe to store metrics and parameters
  df_metrics = pd.DataFrame(columns=['f1', 'auc', 'accuracy', 'precision', 'conf_matrix', 'TP', 'TN', 'FP', 'FN'])

  fig, axs = plt.subplots(1, k, figsize=(20, 4))

  for i, (train_ids, test_ids) in enumerate(kf.split(X, y)):
      X_train = X.iloc[train_ids]
      X_test = X.iloc[test_ids]
      y_train = y.iloc[train_ids]
      y_test = y.iloc[test_ids]

      # Processing GridSearchCV for hyperparameter tuning
      grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=['roc_auc', 'f1'], refit='roc_auc', verbose=0)
      grid_search.fit(X_train, y_train)

      # Getting the best parameters
      best_params = grid_search.best_params_
      print("Best Parameters:", best_params)

      # Using the best model from GridSearchCV
      best_model = grid_search.best_estimator_

      # Fitting the best model on the training data
      history = best_model.fit(X_train, y_train)

      # Predicting probabilities
      y_pred_proba = best_model.predict_proba(X_test)[:, 1]

      # Calculating AUC score
      roc_auc = roc_auc_score(y_test, y_pred_proba)
      print("AUC Score:", roc_auc)
      # Calculating F1 score
      y_pred = best_model.predict(X_test)
      f1 = metrics.f1_score(y_test, y_pred, average='weighted')
      print("F1 Score:", f1)
      # Calculating accuracy
      accuracy = accuracy_score(y_test, y_pred)
      # Calculating precision
      precision = metrics.precision_score(y_test, y_pred, average='weighted')

      # Confusion Matrix
      conf_matrix = confusion_matrix(y_test, y_pred)
      sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', ax=axs[i % k])
      axs[i % k].set_xlabel('Predicted Label')
      axs[i % k].set_ylabel('True Label')
      axs[i % k].set_title(f'Fold {i+1} - Confusion Matrix')
      axs[i % k].axis('equal')

      print("-" * 50)

      # Saving metrics
      row = pd.Series()
      row['f1'] = f1
      row['auc'] = roc_auc
      row['accuracy'] = accuracy
      row['precision'] = precision
      row['conf_matrix'] = conf_matrix
      row['TP'] = conf_matrix[1][1]
      row['TN'] = conf_matrix[0][0]
      row['FP'] = conf_matrix[0][1]
      row['FN'] = conf_matrix[1][0]
      df_metrics = pd.concat([df_metrics, pd.DataFrame(row).transpose()]).reset_index(drop=True)
      
  return df_metrics

def print_statistics(df_metrics):
  print(df_metrics[['f1','auc','accuracy','precision']].mean())
  average_conf_matrix = df_metrics['conf_matrix'].sum() / len(df_metrics)

  plt.figure(figsize=(4, 4))
  sns.heatmap(average_conf_matrix, annot=True, cmap='Blues', fmt='g')
  plt.gca().set_aspect('equal')
  plt.gca().set_xlabel('Predicted Label')
  plt.gca().set_ylabel('True Label')
  plt.gca().set_title(f'Average Confusion Matrix')
  plt.gca().axis('equal');