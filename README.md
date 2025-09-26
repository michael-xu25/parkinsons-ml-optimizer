# Parkinson's Disease ML Performance Optimizer

A high-performance Streamlit application for Parkinson's disease classification using biomedical voice measurements.

## Features

- **85/5/10 Data Split**: Optimal train/validation/test split for robust evaluation
- **Self-contained ML Implementation**: Custom algorithms without external dependencies
  - Logistic Regression with gradient descent
  - K-Nearest Neighbors with euclidean distance
  - Naive Bayes with Gaussian distribution
- **Feature Engineering**: Correlation-based selection and z-score normalization
- **Performance Optimization**: Achieved 97.14% F1-Score on test set

## Results Achieved

- **Best Model**: K-Nearest Neighbors
- **Test Set Performance**:
  - Accuracy: 94.74%
  - Precision: 94.44%
  - Recall: 100.00%
  - F1-Score: 97.14%

## Usage

1. Install Streamlit: `pip install streamlit pandas numpy`
2. Run the application: `streamlit run app.py`
3. Click "Load Parkinson's Dataset"
4. Configure feature selection and scaling options
5. Click "🎯 Start Performance Optimization"
6. View results and model performance metrics

## Dataset

Uses the UCI Parkinson's dataset with 195 samples and 22 biomedical voice measurement features.
Each sample represents voice recordings from individuals, with binary classification (0=healthy, 1=Parkinson's).

## Technical Implementation

- **Data Processing**: Custom train/test split with stratification
- **Feature Selection**: Correlation-based ranking with configurable k-best selection
- **Scaling**: Z-score normalization using training statistics
- **Model Training**: Grid search hyperparameter optimization
- **Evaluation**: Comprehensive metrics including confusion matrix analysis
