import streamlit as st
import pandas as pd
import numpy as np
import time
import math
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Parkinson's Disease ML Performance Optimizer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Parkinson's Disease ML Performance Optimizer")
st.markdown("**Focus: Maximum Classification Performance with 85/5/10 Split**")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'results' not in st.session_state:
    st.session_state.results = {}

def load_data():
    """Load the Parkinson's dataset"""
    try:
        # Load the attached dataset
        data = pd.read_csv('attached_assets/parkinsons_1758889282656.data 2')
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def prepare_data(data):
    """Prepare features and target from the dataset"""
    # Remove name column, keep status as target
    X = data.drop(['name', 'status'], axis=1)
    y = data['status']
    return X, y

def train_test_split_custom(X, y, test_size=0.1, random_state=42):
    """Custom train-test split implementation"""
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return (X.iloc[train_indices], X.iloc[test_indices], 
            y.iloc[train_indices], y.iloc[test_indices])

def split_data(X, y):
    """Split data into 85% train, 5% validation, 10% test"""
    # First split: separate test set (10%)
    X_temp, X_test, y_temp, y_test = train_test_split_custom(X, y, test_size=0.10, random_state=42)
    
    # Second split: separate train (85%) and validation (5%) from remaining 90%
    # 5/90 = 0.0556 to get 5% of total data
    val_size_relative = 0.05 / 0.90
    X_train, X_val, y_train, y_val = train_test_split_custom(X_temp, y_temp, test_size=val_size_relative, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def standardize_features(X_train, X_val, X_test):
    """Standardize features using z-score normalization"""
    # Calculate mean and std from training data
    mean = X_train.mean()
    std = X_train.std()
    
    # Apply standardization
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_val_scaled, X_test_scaled

class SimpleLogisticRegression:
    """Simple logistic regression implementation"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model"""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Gradient descent
        for i in range(self.max_iter):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute cost
            cost = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Early stopping if cost is very low
            if cost < 1e-6:
                break
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_pred)
        return (predictions >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X = np.array(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

class SimpleKNN:
    """Simple k-nearest neighbors implementation"""
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def euclidean_distance(self, x1, x2):
        """Calculate euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Majority vote
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)
        
        return np.array(predictions)

class SimpleNaiveBayes:
    """Simple Naive Bayes implementation"""
    
    def __init__(self):
        self.class_priors = {}
        self.feature_means = {}
        self.feature_stds = {}
        self.classes = None
    
    def fit(self, X, y):
        """Train the Naive Bayes model"""
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        
        for cls in self.classes:
            # Calculate class prior
            self.class_priors[cls] = np.mean(y == cls)
            
            # Calculate feature statistics for this class
            X_cls = X[y == cls]
            self.feature_means[cls] = np.mean(X_cls, axis=0)
            self.feature_stds[cls] = np.std(X_cls, axis=0) + 1e-6  # Add small value to avoid division by zero
    
    def gaussian_pdf(self, x, mean, std):
        """Calculate Gaussian probability density function"""
        exponent = -0.5 * ((x - mean) / std) ** 2
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            class_scores = {}
            
            for cls in self.classes:
                # Calculate likelihood
                likelihood = np.prod(self.gaussian_pdf(x, self.feature_means[cls], self.feature_stds[cls]))
                
                # Calculate posterior (prior * likelihood)
                class_scores[cls] = self.class_priors[cls] * likelihood
            
            # Predict class with highest score
            prediction = max(class_scores.keys(), key=lambda k: class_scores[k])
            predictions.append(prediction)
        
        return np.array(predictions)

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    accuracy = np.mean(y_true == y_pred)
    
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def select_best_features(X_train, y_train, k=15):
    """Simple feature selection using correlation with target"""
    correlations = {}
    
    for column in X_train.columns:
        # Calculate correlation with target
        correlation = abs(np.corrcoef(X_train[column], y_train)[0, 1])
        if not np.isnan(correlation):
            correlations[column] = correlation
    
    # Select top k features
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in sorted_features[:k]]
    
    return selected_features

# Main application flow
st.header("🚀 Performance Optimization Pipeline")

# Step 1: Data Loading
st.subheader("1. Data Loading")
if st.button("Load Parkinson's Dataset"):
    with st.spinner("Loading dataset..."):
        data = load_data()
        if data is not None:
            st.session_state.data = data
            st.session_state.data_loaded = True
            
            # Display basic info
            st.success(f"✅ Dataset loaded successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(data))
            with col2:
                st.metric("Features", len(data.columns) - 2)
            with col3:
                parkinson_count = data['status'].sum()
                st.metric("Parkinson's Cases", f"{parkinson_count} ({parkinson_count/len(data)*100:.1f}%)")

# Step 2: Model Training and Optimization
if st.session_state.data_loaded:
    st.subheader("2. Model Training & Optimization")
    
    col1, col2 = st.columns(2)
    with col1:
        use_feature_selection = st.checkbox("Use Feature Selection", value=True)
        if use_feature_selection:
            n_features = st.slider("Number of features", 5, 20, 15)
    
    with col2:
        use_scaling = st.checkbox("Use Feature Scaling", value=True)
    
    if st.button("🎯 Start Performance Optimization", type="primary"):
        data = st.session_state.data
        X, y = prepare_data(data)
        
        # Split data (85/5/10)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        st.info(f"Data split: Train={len(X_train)} (85%), Val={len(X_val)} (5%), Test={len(X_test)} (10%)")
        
        # Feature selection
        if use_feature_selection:
            with st.spinner("Performing feature selection..."):
                selected_features = select_best_features(X_train, y_train, n_features)
                X_train = X_train[selected_features]
                X_val = X_val[selected_features]
                X_test = X_test[selected_features]
                st.success(f"✅ Selected {len(selected_features)} features: {', '.join(selected_features[:5])}...")
        
        # Feature scaling
        if use_scaling:
            with st.spinner("Scaling features..."):
                X_train, X_val, X_test = standardize_features(X_train, X_val, X_test)
                st.success("✅ Features standardized")
        
        # Train models
        models = {
            "Logistic Regression": SimpleLogisticRegression(learning_rate=0.01, max_iter=1000),
            "K-Nearest Neighbors": SimpleKNN(k=5),
            "Naive Bayes": SimpleNaiveBayes()
        }
        
        results = {}
        progress_bar = st.progress(0)
        
        for i, (model_name, model) in enumerate(models.items()):
            with st.spinner(f"Training {model_name}..."):
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_pred = model.predict(X_val)
                val_metrics = calculate_metrics(y_val, val_pred)
                
                # Evaluate on test set
                test_pred = model.predict(X_test)
                test_metrics = calculate_metrics(y_test, test_pred)
                
                training_time = time.time() - start_time
                
                results[model_name] = {
                    'model': model,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'training_time': training_time
                }
                
                progress_bar.progress((i + 1) / len(models))
        
        st.session_state.results = results
        
        # Display results
        st.subheader("🏆 Performance Results")
        
        # Create results table
        results_data = []
        for model_name, result in results.items():
            test_metrics = result['test_metrics']
            val_metrics = result['val_metrics']
            results_data.append({
                'Model': model_name,
                'Test Accuracy': f"{test_metrics['accuracy']:.4f}",
                'Test Precision': f"{test_metrics['precision']:.4f}",
                'Test Recall': f"{test_metrics['recall']:.4f}",
                'Test F1-Score': f"{test_metrics['f1_score']:.4f}",
                'Val F1-Score': f"{val_metrics['f1_score']:.4f}",
                'Training Time': f"{result['training_time']:.2f}s"
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Find best model
        best_model_name = max(results.keys(), 
                            key=lambda x: results[x]['test_metrics']['f1_score'])
        best_f1 = results[best_model_name]['test_metrics']['f1_score']
        
        st.success(f"🥇 **Best Model: {best_model_name}** with F1-Score: {best_f1:.4f}")
        
        # Display best model details
        best_result = results[best_model_name]
        st.subheader(f"Best Model Details: {best_model_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Test Set Performance:**")
            test_metrics = best_result['test_metrics']
            st.write(f"- Accuracy: {test_metrics['accuracy']:.4f}")
            st.write(f"- Precision: {test_metrics['precision']:.4f}")
            st.write(f"- Recall: {test_metrics['recall']:.4f}")
            st.write(f"- F1-Score: {test_metrics['f1_score']:.4f}")
        
        with col2:
            st.write("**Confusion Matrix:**")
            st.write(f"- True Positives: {test_metrics['tp']}")
            st.write(f"- True Negatives: {test_metrics['tn']}")
            st.write(f"- False Positives: {test_metrics['fp']}")
            st.write(f"- False Negatives: {test_metrics['fn']}")
        
        # Performance analysis
        st.subheader("📊 Performance Analysis")
        
        all_f1_scores = [result['test_metrics']['f1_score'] for result in results.values()]
        all_accuracies = [result['test_metrics']['accuracy'] for result in results.values()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best F1-Score", f"{max(all_f1_scores):.4f}")
        with col2:
            st.metric("Best Accuracy", f"{max(all_accuracies):.4f}")
        with col3:
            avg_f1 = np.mean(all_f1_scores)
            st.metric("Average F1-Score", f"{avg_f1:.4f}")
        
        # Recommendations
        if best_f1 > 0.95:
            st.success("🎉 Excellent performance! The model is ready for deployment.")
        elif best_f1 > 0.90:
            st.info("👍 Very good performance. Consider fine-tuning for production use.")
        elif best_f1 > 0.85:
            st.warning("⚠️ Good performance but room for improvement. Consider feature engineering.")
        else:
            st.error("❌ Performance needs improvement. Consider different approaches or more data.")
        
        # Model comparison insights
        st.subheader("🔍 Model Comparison Insights")
        
        for model_name, result in results.items():
            test_acc = result['test_metrics']['accuracy']
            val_acc = result['val_metrics']['accuracy']
            overfitting = abs(val_acc - test_acc)
            
            if overfitting > 0.1:
                st.warning(f"⚠️ {model_name}: Potential overfitting detected (Val: {val_acc:.3f}, Test: {test_acc:.3f})")
            else:
                st.info(f"✅ {model_name}: Good generalization (Val: {val_acc:.3f}, Test: {test_acc:.3f})")

# Footer
st.markdown("---")
st.markdown("*Self-contained ML implementation optimized for maximum classification performance on Parkinson's disease detection*")