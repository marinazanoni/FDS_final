
# Ignore all warnings
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Nadam, Nadam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.activations import elu
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import f1_score
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
      grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=0)
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
  
# Define the model creation within a function for reusability
def create_model_CNN(X_train):
    model = Sequential([
              Conv1D(filters=128, kernel_size=3, activation=elu, input_shape=(X_train.shape[1], 1)),
              Conv1D(filters=128, kernel_size=3, activation=elu),
              MaxPooling1D(pool_size=2),
              Flatten(),
              Dense(128, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(128, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(128, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(128, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(64, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(64, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(64, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(64, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(1, activation='sigmoid') # Sigmoid because we want probability output between 0 and 1
    ])

    # Choose the optimizer and loss function
    optimizer = Nadam(learning_rate=0.001)
    loss_function = BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['Recall', 'AUC', 'FalseNegatives', 'accuracy'])
    
    return model
  
  # Define the model creation within a function for reusability
def create_model_NN(X_train):
    model = Sequential([
              Dense(128, activation=elu, input_shape=(X_train.shape[1],)),
              BatchNormalization(),
              Dropout(0.03),
              Dense(128, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(128, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(128, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(64, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(64, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(64, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(64, activation=elu),
              BatchNormalization(),
              Dropout(0.03),
              Dense(1, activation='sigmoid') # Sigmoid because we want probability output between 0 and 1
    ])

    # Choose the optimizer and loss function
    optimizer = Nadam(learning_rate=0.001)
    loss_function = BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['Recall', 'AUC', 'FalseNegatives', 'accuracy'])
    
    return model 
  
def CV_NN_stats(X, y, n_splits, class_weight, model_type):
  # Initialize Stratified K-fold
  stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4)


  # Lists to store scores
  auc_scores = []
  f1_scores = []
  fn_rates = []
  accuracy_scores = []
  average_conf_matrix = np.zeros((2, 2))
  for train_index, test_index in stratified_kfold.split(X, y):
      X_train, X_temp = X.iloc[train_index], X.iloc[test_index]
      y_train, y_temp = y.iloc[train_index], y.iloc[test_index]

      # Further split the training data into train and validation sets
      X_test, X_val_fold, y_test, y_val_fold = train_test_split(
          X_temp, y_temp, test_size=0.6, random_state=42
      )

      # Create the model
      if model_type == 'CNN':
        model = create_model_CNN(X_train)
      else:
        model = create_model_NN(X_train)

      # Define a callback to save the best model based on validation loss
      checkpoint = ModelCheckpoint("best_model_fold.keras", monitor='val_loss', save_best_only=True)

      # Add learning rate scheduler
      reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

      # Add early stopping based on validation loss
      early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
      class_weight = class_weight
      # Train the model with the callback using the train and validation subsets
      history = model.fit(X_train, y_train, epochs=150, batch_size=32,
                          validation_data=(X_val_fold, y_val_fold),
                          callbacks=[checkpoint, reduce_lr, early_stop], 
                          class_weight=class_weight,
                          verbose=0)


      # Load the best model for this fold
      best_model = load_model('best_model_fold.keras')

      # Evaluate the model on the test fold
      test_loss, test_recall, test_auc, test_fn, test_acc = best_model.evaluate(X_test, y_test)
      auc_scores.append(test_auc)
      accuracy_scores.append(test_acc)
      
      print("Test Recall:", test_recall)
      print("Test AUC:", test_auc)
      print("Test FN:", test_fn)
      print("Test Accuracy:", test_acc)


      # Predict probabilities on the test set
      y_pred_proba = best_model.predict(X_test)

      # Get binary predictions based on probability threshold 
      y_pred = np.where(y_pred_proba > 0.48, 1, 0)

      # Calculate and print classification report
      print("Classification Report:")
      print(classification_report(y_test, y_pred))

      # Calculate confusion matrix
      conf_matrix = confusion_matrix(y_test, y_pred)

      fn = conf_matrix[1][0]
      fn_rate = fn / (fn + conf_matrix[1][1] + conf_matrix[0][1] + conf_matrix[0][0]) 
      fn_rates.append(fn_rate)

      average_conf_matrix += conf_matrix

      # Plot confusion matrix heatmap
      plt.figure(figsize=(8, 6))
      sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
      plt.xlabel('Predicted Label')
      plt.ylabel('True Label')
      plt.title('Confusion Matrix')
      plt.show()

      # Calculate ROC curve
      fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
      roc_auc = auc(fpr, tpr)

      # Plot ROC curve
      plt.figure(figsize=(8, 6))
      plt.plot(fpr, tpr, label=f'ROC Curve (Area = {roc_auc:.2f})')
      plt.plot([0, 1], [0, 1], 'k--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic (ROC) Curve')
      plt.legend()
      plt.show()


      # Plot training and validation loss
      plt.figure(figsize=(8, 6))
      plt.plot(history.history['loss'], label='Training Loss')
      plt.plot(history.history['val_loss'], label='Validation Loss')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.title('Training and Validation Loss')
      plt.legend()
      plt.show()

      f1_overall = f1_score(y_test, y_pred, average='weighted')
      f1_scores.append(f1_overall)

  # Print mean and standard deviation of accuracy scores across folds
  print(f'Mean AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})')
  print(f'Mean F1 Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})')
  print(f'Mean FN Rate: {np.mean(fn_rates):.4f} (+/- {np.std(fn_rates):.4f})')
  print(f'Mean Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})')

  average_conf_matrix /= 5

  # Plot the average confusion matrix
  plt.figure(figsize=(8, 6))
  sns.heatmap(average_conf_matrix, annot=True, cmap='Blues', fmt='g')
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.title('Average Confusion Matrix Across Folds')
  plt.show()