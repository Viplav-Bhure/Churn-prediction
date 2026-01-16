# E-Commerce Customer Churn Prediction

This project predicts whether a customer is likely to churn (stop doing business) based on their purchasing behavior, demographics, and satisfaction metrics. The dataset is sourced from the Kaggle E-Commerce Customer Churn Analysis and Prediction dataset.

## Project Structure

- `app.py`: Streamlit web application that takes user input and displays churn probability.
- `main.py`: Complete pipeline script that loads data, performs EDA, trains multiple models, and saves the best one.
- `requirements.txt`: List of Python dependencies required to run the project.
- `final_churn_model.pkl`: The best-performing trained machine learning model (saved automatically).
- `cat_options.pkl`: Dictionary of unique values for categorical features (used for dropdowns in the app).
- `eda_correlation_matrix.png`: Generated heatmap visualization of feature correlations.

## Features Used

The model uses the following key features to predict churn:
- **Demographics:** Gender, Marital Status, City Tier.
- **Account Info:** Tenure, Preferred Login Device, Payment Mode.
- **Behavior:** Days Since Last Order, Cashback Amount, Hours Spent on App.
- **Logistics:** Distance from Warehouse to Home.
- **Engagement:** Satisfaction Score, Complain History, Order Count, Coupons Used.


## Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone [https://github.com/Viplav-Bhure/Churn-prediction.git](https://github.com/Viplav-Bhure/Churn-prediction.git)
    cd Churn-prediction
    ```

2. **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\Activate

    # Mac/Linux
    source venv/bin/activate
    ```

3. **Install necessary packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Train the model (First Run Only):**
   *Note: You must run this script first to generate the model files.*
   *Ensure `churn_data.xlsx` is in the project folder.*
    ```bash
    python main.py
    ```

5. **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

6. Open the URL displayed in the terminal (usually `http://localhost:8501`) in your web browser.

## Usage

- **Training:** Run `main.py` to process the raw data, generate an EDA report (saved as an image), and retrain the models.
- **Prediction:** Use the Streamlit web interface to adjust customer parameters (like Tenure, Satisfaction Score, etc.) and click "Predict Churn Risk" to see if the customer is Safe or at High Risk.

## How It Works

1. **Data Pipeline:** `main.py` loads the Excel dataset and cleans column names.
2. **Preprocessing:**
   - Numerical values are imputed with the Median and Scaled.
   - Categorical values are imputed with 'missing' and One-Hot Encoded.
3. **Model Selection:** The script trains four different models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) and automatically saves the one with the highest accuracy.
4. **Deployment:** `app.py` loads this saved pipeline and uses it to inference on live user input.

## Exploratory Data Analysis (EDA)

The `main.py` script automatically performs:
- **Missing Value Analysis:** Checks and reports null values.
- **Correlation Mapping:** Generates `eda_correlation_matrix.png` to show relationships between numerical features.
- **Class Balance:** Checks the percentage of Churn vs. Non-Churn customers.

## Models Trained

The following algorithms are compared during training:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier (Often selected as the best performer)

## Acknowledgements

- **Dataset:** [E-Commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) by Ankit Verma on Kaggle.
- **Tools:** Streamlit for the web framework, Scikit-Learn and XGBoost for machine learning.