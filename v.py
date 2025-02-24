import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Streamlit Page Configuration
st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

# Database Connection
server = "LAPTOP-DBNSDTGH"
database = "SKYWARD_ProcomOfficeSolutions_20160816074850680"
engine = create_engine(f"mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server")

# Load Data
query = """
SELECT TOP (1000) UniqueID, Probability, Amount, StageID, CustomerTypeID, IndustryID, LeadSourceID, OpportunitySalesStageID
FROM [dbo].[OPP_Opportunities]
"""
df = pd.read_sql(query, engine)

# Check if Data is Empty
if df.empty:
    st.error("No data found in the database. Please check your SQL connection and query.")
    st.stop()

# Handle Missing Values
def fill_with_mode_or_default(series, default):
    """Fill missing values with mode or default value."""
    mode = series.mode()
    return mode[0] if not mode.empty else default

df.fillna({
    'Probability': df['Probability'].mean(),
    'Amount': df['Amount'].mean(),
    'StageID': fill_with_mode_or_default(df['StageID'], 0),
    'CustomerTypeID': fill_with_mode_or_default(df['CustomerTypeID'], 0),
    'IndustryID': fill_with_mode_or_default(df['IndustryID'], 0),
    'LeadSourceID': fill_with_mode_or_default(df['LeadSourceID'], 0),
    'OpportunitySalesStageID': fill_with_mode_or_default(df['OpportunitySalesStageID'], 0)
}, inplace=True)

# Encode Categorical Data
for col in ['StageID', 'CustomerTypeID', 'IndustryID', 'LeadSourceID', 'OpportunitySalesStageID']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define Risk Categories
def classify_risk(prob):
    """Classify risk based on probability."""
    if prob < 40:
        return "High Risk"
    elif 40 <= prob <= 70:
        return "Medium Risk"
    else:
        return "Low Risk"

df['ChurnRisk'] = df['Probability'].apply(classify_risk)

# Prepare Data for ML
X = df.drop(columns=['UniqueID', 'ChurnRisk'])
y = df['ChurnRisk']

# Ensure Data is Not Empty Before Splitting
if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check for missing values or filtering issues.")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train ML Model with Cross-Validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)


# Fit the model on the training data
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


# Streamlit Dashboard
st.title("📊 Customer Churn Prediction Dashboard")

# Pie Chart for Churn Risk Distribution
st.subheader("Churn Risk Distribution")
churn_counts = df['ChurnRisk'].value_counts().reset_index()
churn_counts.columns = ['ChurnRisk', 'Count']

fig = px.pie(churn_counts, values='Count', names='ChurnRisk', color_discrete_sequence=['green', 'red', 'orange'])
fig.update_layout(title_text="Customer Churn Risk Distribution")
st.plotly_chart(fig, use_container_width=True)

# Display Cluster Summary
st.subheader("🔍 Cluster Analysis & Recommendations")
selected_cluster = st.selectbox("Select a Cluster", df['ChurnRisk'].unique())

cluster_data = df[df['ChurnRisk'] == selected_cluster]

st.write(f"**Churn Risk Distribution in Cluster {selected_cluster}:**")
st.bar_chart(cluster_data['ChurnRisk'].value_counts())

# Recommendations Based on Risk Level
st.subheader("📌 Recommended Actions Based on Churn Risk")
if selected_cluster == "High Risk":
    st.warning("🔴 **High Risk Customers**: Take Immediate Actions!")
    st.write("- **Personalized Retention Offers**: Tailor offers to individual needs.")
    st.write("- **Priority Support**: Ensure timely and effective support.")
    st.write("- **Feedback Collection**: Gather insights to improve services.")
    st.write("- **Engagement Campaigns**: Regularly engage with customers.")
elif selected_cluster == "Medium Risk":
    st.info("🟠 **Medium Risk Customers**: Keep Them Engaged!")
    st.write("- **Follow-Up Calls**: Regular check-ins to ensure satisfaction.")
    st.write("- **Regular Updates**: Keep customers informed about new developments.")
    st.write("- **Loyalty Programs**: Reward loyalty to encourage retention.")
    st.write("- **Educational Content**: Provide valuable insights and tips.")
else:
    st.success("🟢 **Low Risk Customers**: Maintain Good Relationships!")
    st.write("- **Maintain Engagement**: Regularly interact with customers.")
    st.write("- **Referral Programs**: Encourage referrals through incentives.")
    st.write("- **Exclusive Offers**: Provide special offers to loyal customers.")
    st.write("- **Customer Appreciation**: Show appreciation through events or gifts.")

st.markdown("📢 *Use these insights to reduce customer churn and improve business growth!* 🚀")