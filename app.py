import streamlit as st
import pandas as pd
import joblib

# CONFIGURATION
st.set_page_config(
    page_title="Churn Prediction AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        height: 3em;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
    h1 {
        text-align: center;
        color: #2c3e50;
    }
    h3 {
        color: #3498db;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        font-size: 24px;
        font-weight: bold;
    }
    .safe {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# LOAD RESOURCES
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('final_churn_model.pkl')
        options = joblib.load('cat_options.pkl')
        return model, options
    except FileNotFoundError:
        return None, None

model, options = load_resources()

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149663.png", width=100)
    st.title("Admin Panel")
    st.info("This tool uses a Machine Learning model XGBoost to predict customer retention risk.")
    st.markdown("---")
    st.markdown("**User Guide:**")
    st.markdown("1. Enter customer details in the main form.")
    st.markdown("2. Click 'Analyze Risk'.")
    st.markdown("3. Get instant prediction.")
    st.markdown("---")
    st.caption("v1.0 | Built by Viplav Bhure")

# MAIN
if model is None:
    st.error("❌ Model files not found! Please run `train_model.py` first to generate the `.pkl` files.")
    st.stop()

st.title("🔮 E-Commerce Customer Churn Prediction")
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Enter customer data below to generate a real-time risk assessment.</p>", unsafe_allow_html=True)
st.markdown("---")

# Form
with st.form("churn_form"):
    
    c1, c2, c3 = st.columns(3)
    
    # COLUMN 1: DEMOGRAPHICS & PROFILE
    with c1:
        st.markdown("### 👤 User Profile")
        gender = st.selectbox("Gender", options['Gender'])
        marital = st.selectbox("Marital Status", options['MaritalStatus'])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
        warehouse = st.number_input("Dist. to Warehouse (km)", min_value=0, value=10)
        num_address = st.number_input("Registered Addresses", 1, 20, 2)

    # COLUMN 2: ACCOUNT & PAYMENT
    with c2:
        st.markdown("### 💳 Account Info")
        login_device = st.selectbox("Login Device", options['PreferredLoginDevice'])
        payment_mode = st.selectbox("Payment Method", options['PreferredPaymentMode'])
        order_cat = st.selectbox("Preferred Order Category", options['PreferedOrderCat'])
        complain = st.selectbox("Any Recent Complaints?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        coupon_used = st.number_input("Coupons Used", 0, 20, 1)
        cashback = st.number_input("Avg Cashback ($)", 0.0, 1000.0, 150.0)

    # COLUMN 3: BEHAVIOR & METRICS
    with c3:
        st.markdown("### 📊 Behavior Stats")
        day_since_order = st.number_input("Days Since Last Order", 0, 365, 5)
        order_count = st.number_input("Total Orders", 1, 500, 5)
        order_hike = st.number_input("Order Hike from Last Year (%)", 0.0, 100.0, 15.0)
        satisfaction = st.slider("Satisfaction Score (1 = High, 5 = Low)", 1, 5, 3)
        app_hours = st.slider("Daily App Usage (Hours)", 0.0, 6.0, 2.0)
        num_devices = st.slider("Registered Devices", 1, 10, 3)

    st.markdown("---")
    
    # Submit Button
    submitted = st.form_submit_button("🚀 Analyze Risk")

# PREDICTION LOGIC
if submitted:
    # Constructing DataFrame
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'PreferredLoginDevice': [login_device],
        'CityTier': [city_tier],
        'WarehouseToHome': [warehouse],
        'PreferredPaymentMode': [payment_mode],
        'Gender': [gender],
        'HourSpendOnApp': [app_hours],
        'NumberOfDeviceRegistered': [num_devices],
        'PreferedOrderCat': [order_cat],
        'SatisfactionScore': [satisfaction],
        'MaritalStatus': [marital],
        'NumberOfAddress': [num_address],
        'Complain': [complain],
        'OrderAmountHikeFromlastYear': [order_hike],
        'CouponUsed': [coupon_used],
        'OrderCount': [order_count],
        'DaySinceLastOrder': [day_since_order],
        'CashbackAmount': [cashback]
    })

    # Get Prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # DISPLAY RESULTS
    st.markdown("### Analysis Result")
    
    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-box danger">
                ⚠️ HIGH CHURN RISK<br>
                <span style="font-size: 18px; font-weight: normal;">Probability of churn: {prob:.2%}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.warning("Action Recommended: Send retention offer or discount coupon immediately.")
    else:
        st.markdown(
            f"""
            <div class="result-box safe">
                ✅ LOYAL CUSTOMER<br>
                <span style="font-size: 18px; font-weight: normal;">Probability of churn: {prob:.2%}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.balloons()