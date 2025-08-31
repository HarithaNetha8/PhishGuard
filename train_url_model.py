import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def generate_sample_dataset(n_samples=10000):
    """
    Generate a sample phishing dataset with 31 features
    This creates synthetic data for demonstration purposes
    """
    np.random.seed(42)
    
    # Feature names
    feature_names = [
        'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens',
        'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore',
        'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon',
        'nb_comma', 'nb_semicolon', 'nb_dollar', 'nb_space', 'nb_www',
        'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url',
        'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain'
    ]
    
    data = []
    labels = []
    
    for i in range(n_samples):
        # Generate label (0 = legitimate, 1 = phishing)
        is_phishing = np.random.choice([0, 1], p=[0.7, 0.3])
        
        if is_phishing:
            # Phishing URL characteristics
            features = [
                np.random.normal(120, 40),  # length_url (longer)
                np.random.normal(25, 10),   # length_hostname
                np.random.choice([0, 1], p=[0.8, 0.2]),  # ip (more likely)
                np.random.poisson(8),       # nb_dots (more dots)
                np.random.poisson(5),       # nb_hyphens (more hyphens)
                np.random.poisson(1),       # nb_at
                np.random.poisson(2),       # nb_qm
                np.random.poisson(3),       # nb_and
                np.random.poisson(1),       # nb_or
                np.random.poisson(2),       # nb_eq
                np.random.poisson(2),       # nb_underscore
                np.random.poisson(1),       # nb_tilde
                np.random.poisson(2),       # nb_percent
                np.random.poisson(6),       # nb_slash
                np.random.poisson(1),       # nb_star
                np.random.poisson(3),       # nb_colon
                np.random.poisson(1),       # nb_comma
                np.random.poisson(1),       # nb_semicolon
                np.random.poisson(1),       # nb_dollar
                np.random.poisson(0.5),     # nb_space
                np.random.poisson(1),       # nb_www
                np.random.poisson(2),       # nb_com
                np.random.poisson(1),       # nb_dslash
                np.random.choice([0, 1], p=[0.7, 0.3]),  # http_in_path
                np.random.choice([0, 1], p=[0.8, 0.2]),  # https_token
                np.random.uniform(0.1, 0.4),  # ratio_digits_url
                np.random.uniform(0.1, 0.3),  # ratio_digits_host
                np.random.choice([0, 1], p=[0.9, 0.1]),  # punycode
                np.random.choice([0, 1], p=[0.8, 0.2]),  # port
                np.random.choice([0, 1], p=[0.9, 0.1]),  # tld_in_path
                np.random.choice([0, 1], p=[0.95, 0.05]) # tld_in_subdomain
            ]
        else:
            # Legitimate URL characteristics
            features = [
                np.random.normal(60, 20),   # length_url (shorter)
                np.random.normal(15, 5),    # length_hostname
                np.random.choice([0, 1], p=[0.95, 0.05]),  # ip (less likely)
                np.random.poisson(3),       # nb_dots (fewer dots)
                np.random.poisson(1),       # nb_hyphens (fewer hyphens)
                np.random.poisson(0.1),     # nb_at
                np.random.poisson(0.5),     # nb_qm
                np.random.poisson(1),       # nb_and
                np.random.poisson(0.1),     # nb_or
                np.random.poisson(0.5),     # nb_eq
                np.random.poisson(0.5),     # nb_underscore
                np.random.poisson(0.1),     # nb_tilde
                np.random.poisson(0.2),     # nb_percent
                np.random.poisson(3),       # nb_slash
                np.random.poisson(0.1),     # nb_star
                np.random.poisson(2),       # nb_colon
                np.random.poisson(0.1),     # nb_comma
                np.random.poisson(0.1),     # nb_semicolon
                np.random.poisson(0.1),     # nb_dollar
                np.random.poisson(0.1),     # nb_space
                np.random.poisson(0.8),     # nb_www
                np.random.poisson(1),       # nb_com
                np.random.poisson(1),       # nb_dslash
                np.random.choice([0, 1], p=[0.9, 0.1]),  # http_in_path
                np.random.choice([0, 1], p=[0.95, 0.05]), # https_token
                np.random.uniform(0.0, 0.2),  # ratio_digits_url
                np.random.uniform(0.0, 0.1),  # ratio_digits_host
                np.random.choice([0, 1], p=[0.99, 0.01]), # punycode
                np.random.choice([0, 1], p=[0.95, 0.05]), # port
                np.random.choice([0, 1], p=[0.99, 0.01]), # tld_in_path
                np.random.choice([0, 1], p=[0.99, 0.01])  # tld_in_subdomain
            ]
        
        # Ensure non-negative values and proper bounds
        features = [max(0, f) for f in features]
        
        data.append(features)
        labels.append(is_phishing)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['label'] = labels
    
    return df

def load_or_generate_dataset():
    """
    Load existing dataset or generate sample data
    """
    # Try to load existing dataset
    dataset_path = 'phishing_dataset.csv'
    
    if os.path.exists(dataset_path):
        logging.info(f"Loading existing dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        logging.info("Generating sample dataset...")
        df = generate_sample_dataset()
        # Save generated dataset
        df.to_csv(dataset_path, index=False)
        logging.info(f"Sample dataset saved to {dataset_path}")
    
    return df

def train_random_forest_model():
    """
    Train Random Forest model for URL-based phishing detection
    """
    # Load dataset
    df = load_or_generate_dataset()
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns].values
    y = df['label'].values
    
    logging.info(f"Dataset shape: {X.shape}")
    logging.info(f"Phishing samples: {np.sum(y)}, Legitimate samples: {len(y) - np.sum(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    logging.info("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Model Accuracy: {accuracy:.4f}")
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info("\nTop 10 Most Important Features:")
    logging.info(feature_importance.head(10))
    
    # Create a pipeline with scaler and model
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', rf_model)
    ])
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/random_forest_model.pkl'
    joblib.dump(pipeline, model_path)
    logging.info(f"Model saved to {model_path}")
    
    return pipeline, accuracy

def main():
    """Main function to train the URL-based phishing detection model"""
    try:
        model, accuracy = train_random_forest_model()
        print(f"URL model training completed successfully! Accuracy: {accuracy:.4f}")
        return True
    except Exception as e:
        logging.error(f"Error training URL model: {str(e)}")
        return False

if __name__ == "__main__":
    main()
