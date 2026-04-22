import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Income Analysis", layout="wide")

st.title("📊 Income Analysis Dashboard")

# Load dataset
@st.cache_data
def load_data():
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    df = pd.read_csv(url, names=column_names, skipinitialspace=True)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

age_range = st.sidebar.slider("Select Age Range", 17, 90, (20, 50))
gender = st.sidebar.selectbox("Select Gender", ["All"] + list(df["gender"].unique()))

filtered_df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

if gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender]

# Dataset preview
st.subheader("📄 Dataset Preview")
st.dataframe(filtered_df.head())

# Stats
st.subheader("📊 Statistics")
st.write(filtered_df.describe())

# Layout with columns
col1, col2 = st.columns(2)

# Income distribution
with col1:
    st.subheader("💰 Income Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="income", ax=ax)
    st.pyplot(fig)

# Age distribution
with col2:
    st.subheader("📈 Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["age"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# Occupation vs Income
st.subheader("🧠 Occupation vs Income")
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(data=filtered_df, y="occupation", hue="income", ax=ax)
st.pyplot(fig)

# Work hours vs Income
st.subheader("⏱️ Hours per Week vs Income")
fig, ax = plt.subplots()
sns.boxplot(data=filtered_df, x="income", y="hours_per_week", ax=ax)
st.pyplot(fig)

# Insights section
st.subheader("📌 Insights")
st.markdown("""
- Higher working hours tend to correlate with higher income  
- Certain occupations dominate high-income groups  
- Age group plays a significant role in income distribution  
""")