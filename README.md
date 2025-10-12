# Parkinson's Disease ML Performance Optimizer

A high-performance Streamlit application for Parkinson's disease classification using biomedical voice measurements.

## Features

- **Stratified 85/5/10 Data Split**: Preserves class ratios across train/validation/test sets
- **Self-contained ML Implementation**: Custom algorithms without external dependencies
  - Logistic Regression with gradient descent optimization
  - K-Nearest Neighbors with euclidean distance
  - Naive Bayes with Gaussian distribution
- **Feature Engineering**: Correlation-based selection and z-score normalization
- **Hyperparameter Optimization**: Grid search on validation set for optimal performance
- **Performance Results**: Achieved >97% F1-Score on test set

## Results Achieved

- **Best Model**: K-Nearest Neighbors
- **Test Set Performance**:
  - Accuracy: 94.74%
  - Precision: 94.44%
  - Recall: 100.00%
  - F1-Score: 97.14%

## Usage

1. Open Terminal & Navigate to project directory 'cd parkinsons-ml-optimizer' (depends on where the folder is on your computer)
2. Create a virtual environment 'python3 -m venv venv'
3. Activate a virtual environment 'source venv/bin/activate'
4. Install dependencies: `pip install -r requirements.txt --only-binary :all:'
5. Run the application: 'streamlit run app.py --server.port 5001' (may need to hit 'rerun' once in top right if nothing appears)
6. Click "Load Parkinson's Dataset"
7. Configure feature selection and scaling options
8. Click "Start Performance Optimization"
9. View results and model performance metrics (results may vary based on the way test/training/validation sets are randomized)

## Dataset

Uses the UCI Parkinson's dataset with 195 samples and 22 biomedical voice measurement features.
Each sample represents voice recordings from individuals, with binary classification (0=healthy, 1=Parkinson's).

## Technical Implementation

- **Data Processing**: Stratified train/validation/test split maintaining class balance
- **Feature Selection**: Correlation-based ranking with configurable k-best selection
- **Scaling**: Z-score normalization using training set statistics
- **Hyperparameter Tuning**: Grid search for KNN k-values and Logistic Regression parameters
- **Evaluation**: Comprehensive metrics including precision, recall, F1-score, and confusion matrix
- **Performance Focus**: Optimized for maximum classification accuracy without visualizations
