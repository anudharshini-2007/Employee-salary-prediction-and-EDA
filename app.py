%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("🚀 Complete Mini-Project: Employee Salary Prediction & EDA Dashboard")
st.write("This application includes data cleaning, exploratory data analysis, visualizations, and an integrated Machine Learning model.")

st.divider()

# 1. Load and Inspect Dataset (Task 1 & 2)
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 200
    experience = np.random.rand(n) * 10  # 0 to 10 years
    test_score = np.random.randint(50, 100, n)
    hours_per_week = np.random.randint(35, 60, n)
    # Salary formula + noise
    salary = 30 + (experience * 12) + (test_score * 0.5) + (hours_per_week * 0.3) + np.random.normal(0, 3, n)

    df = pd.DataFrame({
        'Experience_Years': np.round(experience, 1),
        'Test_Score': test_score,
        'Hours_Per_Week': hours_per_week,
        'Salary_USD': np.round(salary, 2)
    })
    return df

df = load_data()

st.header("1. Dataset & Preprocessing")
if st.checkbox("Show Raw Dataset"):
    st.write(df.head(10))

# Data Cleaning simulation (dropping nulls if any)
df_cleaned = df.dropna()
st.success(f"Data cleaned & preprocessed successfully! Total Rows: {len(df_cleaned)}, Columns: {len(df_cleaned.columns)}")

st.divider()

# 2. Exploratory Data Analysis & Visualizations (Task 3 & 4)
st.header("2. Exploratory Data Analysis & Visualizations")
st.write("Summary Statistics of Cleaned Dataset:")
st.write(df_cleaned.describe())

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Experience vs Salary (Scatter Chart)**")
    st.scatter_chart(df_cleaned, x='Experience_Years', y='Salary_USD')

with col2:
    st.markdown("**Test Score vs Salary (Scatter Chart)**")
    st.scatter_chart(df_cleaned, x='Test_Score', y='Salary_USD')

st.divider()

# 3. Train and Integrate ML Model (Task 5)
st.header("3. Machine Learning Model Training")

X = df_cleaned[['Experience_Years', 'Test_Score', 'Hours_Per_Week']]
y = df_cleaned['Salary_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

m_col1, m_col2 = st.columns(2)
m_col1.metric("Model R² Score (Accuracy)", f"{r2:.2f}")
m_col2.metric("Mean Squared Error", f"{mse:.2f}")

st.divider()

# 4. Streamlit Interface for Live Predictions (Task 6 & 7)
st.header("4. Live Prediction Interface")
st.sidebar.header("🎛️ Input Employee Features")

exp_input = st.sidebar.slider("Experience (Years)", 0.0, 10.0, 3.0, 0.1)
score_input = st.sidebar.slider("Test Score (50-100)", 50, 100, 75)
hours_input = st.sidebar.slider("Hours Per Week", 35, 60, 40)

if st.sidebar.button("Predict Salary"):
    input_data = pd.DataFrame([[exp_input, score_input, hours_input]],
                              columns=['Experience_Years', 'Test_Score', 'Hours_Per_Week'])
    prediction = model.predict(input_data)[0]

    st.subheader("🎯 Prediction Result")
    st.success(f"Predicted Estimated Salary: **${prediction:.2f}k USD per year**")
    st.balloons()
