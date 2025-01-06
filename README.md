# Email Spam Detection

## Description
The Email Spam Detection Project aims to develop a robust system for classifying emails as spam or legitimate (ham). By analyzing the content and characteristics of emails using machine learning techniques, this system provides an effective solution to filter unwanted messages and improve email communication.

## Objective
The goal of this project is to:
- Build a machine learning model to classify emails as spam or ham.
- Evaluate the effectiveness of different classifiers using appropriate metrics.
- Optimize model performance through preprocessing, feature extraction, and hyperparameter tuning.

## Steps Implemented
1. **Preprocessing**: Cleaned and prepared the email dataset for analysis.
2. **Feature Extraction**: Extracted meaningful features from email content.
3. **TF-IDF Vectorization**: Converted text data into numerical form using Term Frequency-Inverse Document Frequency (TF-IDF).
4. **Modeling**: Implemented and evaluated multiple machine learning models.
5. **Evaluation**: Assessed models using metrics like accuracy, precision, recall, and F1 score.
6. **Hyperparameter Tuning**: Fine-tuned the SVM model for optimal performance.
7. **ROC Curve Analysis**: Plotted the ROC curve to evaluate model performance.

## Models and Results
The following machine learning models were evaluated on the dataset:

| Model              | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| **Naive Bayes**     | 0.9721   | 0.9743    | 0.9138 | 0.9431   |
| **Support Vector Machine (SVM)** | 0.9852   | 0.9823    | 0.9586 | 0.9703   |
| **Random Forest**   | 0.9782   | 0.9749    | 0.9379 | 0.9561   |

### Hyperparameter Tuning on SVM
Hyperparameter tuning with `{'C': 10, 'kernel': 'rbf'}` improved the SVM model's performance, achieving:

- **Accuracy**: 0.9878

### ROC Curve
The ROC curve for the best-performing SVM model demonstrated its ability to distinguish between spam and ham emails effectively.

## Conclusion
The **Support Vector Machine (SVM)** emerged as the best-performing model, achieving a high accuracy of approximately 98.78%. The process of hyperparameter tuning significantly enhanced the model's classification capabilities.

While accuracy is a key metric, additional metrics such as precision, recall, and F1 score provide a more comprehensive understanding of the model's performance. The project highlights the importance of preprocessing, feature extraction, and careful parameter selection in building an effective spam detection system.

## Features
- Email content preprocessing and feature extraction.
- Implementation of multiple machine learning classifiers.
- Hyperparameter tuning to optimize model performance.
- Comprehensive evaluation using key metrics and ROC curve analysis.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - Scikit-learn
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn

