# Final Project for the "Foundamentals of Data Science" course (MSc in Data Science) a.a. 2023/2024

# Hearth Disease Prediction

## Abstract

This report proposes a detailed analysis of various machine learning models in order to predict the presence (or absence) of heart disease. The main objective is to evaluate and compare the most popular models in order to identify the most suitable one for early detection of heart attacks.

Six different machine learning models were explored: Logistic Regression, Support Vector Machines (SVM), Linear Discriminant Analysis (LDA), Random Forest, Convolutional Neural Networks (CNN) and Neural Networks (NN). The database on which the models were trained is the [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease). Challenges we faced include the small size of the dataset, the imbalance of some classes (i.e., sex), and the absence of many values. Nevertheless, the features are related to very specific physical medical parameters, and therefore very useful for research, which is the reason why this dataset has been much studied.   

Before proceeding with model training, the dataset was analyzed and preprocessed.
In particular, an analysis was carried out to identify the best interpolation method for the missing values. Subsequently, parameter optimization by cross validation was carried out for each model. Performance evaluation metrics, in our case F1-score, AUC-ROC, accuracy, precision and the false negatives rate were used to compare the performance of the models. In particular we paid attention to the false negatives rate, since it is important to minimize the number of false negatives in order to avoid the risk of not detecting a heart attack, which is way worse than a false positive, since the former can lead to death, the latter to further tests.

From our research it follows that the best models are:
- Logistic Regression, with good overall metrics
- Support Vector Machines, for the lower rate of false negatives, i.e. 2%

## Team Members:
- Giorgio Bertone
- Filippo Parlanti
- Stefano Rinaldi
- Marina Zanoni
