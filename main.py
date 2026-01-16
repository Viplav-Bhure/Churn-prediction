import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load Data & Clean Columns
def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath, sheet_name='E Comm')
    
    # Drop ID as it's not a feature
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    
    # Handle potential whitespace in text columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
        
    return df

# Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("\n--- Starting EDA")
    print(f"Shape: {df.shape}")
    print("\nMissing Values:\n", df.isnull().sum())
    
    # Correlation Matrix Visualization
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig('eda_correlation_matrix.png')
    print("EDA: Correlation matrix saved as 'eda_correlation_matrix.png'")
    
    # Target Distribution
    print("\nTarget Distribution (Churn):")
    print(df['Churn'].value_counts(normalize=True))

# Build Preprocessing Pipeline
def get_pipeline(X):
    # Identify types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    print(f"\nNumeric Features: {list(numeric_features)}")
    print(f"Categorical Features: {list(categorical_features)}")

    # Numeric Transformer: Impute missing with Median, then Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: Impute missing with 'missing', then OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Train & Compare Models
def compare_models(X_train, y_train, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    print("\n--- Model Comparison (Accuracy)")
    for name, model in models.items():
        # Create full pipeline for each model
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        # Cross-validation
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        avg_score = scores.mean()
        print(f"{name}: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_model = clf
            best_model_name = name

    print(f"\n Best Model: {best_model_name} with Accuracy: {best_score:.4f}")
    return best_model

# Main
if __name__ == "__main__":
    # Load
    df = load_data('churn_data.xlsx')
        
    # EDA
    perform_eda(df)
        
    # Split
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Preprocess Setup
    preprocessor = get_pipeline(X)
        
    # Compare & Train Best
    final_model = compare_models(X_train, y_train, preprocessor)
        
    # Final Evaluation
    print("\n Final Evaluation on Test Set")
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    print(classification_report(y_test, y_pred))
        
    # Save
    joblib.dump(final_model, 'final_churn_model.pkl')
    print("Model saved as 'final_churn_model.pkl'")
        
    # Save unique values for categorical columns (for Streamlit)
    cat_columns = X.select_dtypes(include=['object']).columns
    unique_values = {col: X[col].unique().tolist() for col in cat_columns}
    joblib.dump(unique_values, 'cat_options.pkl')
    print("Categorical options saved as 'cat_options.pkl'")
