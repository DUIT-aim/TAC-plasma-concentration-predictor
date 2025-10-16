import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. 加载模型与标准化器
# ===============================
# VotingRegressor 模型 (GradientBoosting : CatBoost : AdaBoost = 2 : 5 : 3)
model = joblib.load('voting_regressor.pkl')

# 训练时使用的标准化器
scaler = joblib.load('scaler.pkl')

# ===============================
# 2. Streamlit 页面标题
# ===============================
st.title("Tacrolimus Plasma Concentration Predictor")

# ===============================
# 3. 定义输入变量
# ===============================
continuous_columns = ['Total_daily_dose','BUN','BMI','ALB','NE','CCR','IBIL','TBIL','Dosing_time']  #分类变量
categorical_columns = ['CYP3A5']  # 分类变量

# 在 Streamlit 界面上创建输入框
st.sidebar.header("Please enter the patient's details")

Total_daily_dose = st.sidebar.number_input("Total daily dose (mg):", min_value=0.5, max_value=10.0, value=5.0)
BUN = st.sidebar.number_input("BUN (mmol/L):", min_value=2.0, max_value=40.0, value=11.5)
BMI = st.sidebar.number_input("BMI (kg/m²):", min_value=15.0, max_value=40.0, value=24.5)
ALB = st.sidebar.number_input("ALB (g/L):", min_value=10.0, max_value=60.0, value=35.0)
NE = st.sidebar.number_input("NE# (10⁹/L):", min_value=0.5, max_value=25.0, value=6.5)
CCR = st.sidebar.number_input("CCR (mL/min):", min_value=15.0, max_value=350.0, value=115.0)
IBIL = st.sidebar.number_input("IBIL (µmol/L):", min_value=0.0, max_value=10.0, value=5.0)
TBIL = st.sidebar.number_input("TBIL (µmol/L):", min_value=0.5, max_value=30.5, value=15.5)
Dosing_time = st.sidebar.number_input("Dosing time (day):", min_value=0.0, max_value=500.0, value=200.0)

# 分类变量 CYP3A5 下拉框
CYP3A5_input = st.sidebar.selectbox(
    "CYP3A5 Genotype:",
    options=["CYP3A5*1*1", "CYP3A5*1*3", "CYP3A5*3*3"]
)
CYP3A5_map = {"CYP3A5*1*1": 1, "CYP3A5*1*3": 2, "CYP3A5*3*3": 3}
CYP3A5 = CYP3A5_map[CYP3A5_input]

# 汇总输入
input_data = np.array([[Total_daily_dose, BUN, BMI, ALB, NE, CCR, IBIL, TBIL, Dosing_time, CYP3A5]])
input_df = pd.DataFrame(input_data, columns=continuous_columns + categorical_columns)

# ===============================
# 4. 标准化输入
# ===============================
input_continuous = input_df[continuous_columns]
input_scaled = scaler.transform(input_continuous)

# 合并标准化连续变量和分类变量
final_input = np.hstack([input_scaled, input_df[categorical_columns].values])

# ===============================
# 5. 模型预测
# ===============================
if st.button("Predict Tacrolimus Plasma Concentration"):
    predicted_value = model.predict(final_input)[0]
    lower_bound = predicted_value * 0.8
    upper_bound = predicted_value * 1.2

    st.subheader("🧪 Predicted Result")
    st.write(f"**Tacrolimus Plasma Concentration = {predicted_value:.2f} ± 20% ng/mL**")
    st.write(f"Estimated range: {lower_bound:.2f} – {upper_bound:.2f} ng/mL")

# ===============================
# 6. SHAP 特征解释
# ===============================
st.subheader("🔍 SHAP Feature Importance Explanation")
try:
    df_train = pd.read_csv('train.csv', encoding='utf-8')
    X_train = df_train[continuous_columns + categorical_columns]
    X_train_scaled = scaler.transform(X_train[continuous_columns])
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=continuous_columns)
    X_train_final_df = pd.concat([X_train_scaled_df, X_train[categorical_columns].reset_index(drop=True)], axis=1)
    input_scaled_df = pd.DataFrame(input_scaled, columns=continuous_columns)
    feature_names = ['Total daily dose','BUN','BMI','ALB','NE#','CCR','IBIL','TBIL','Dosing time','CYP3A5']
    final_input_df = pd.DataFrame(final_input, columns=feature_names)

    explainer = shap.Explainer(model.predict, X_train_final_df)
    shap_values = explainer(final_input_df)

    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("SHAP_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("SHAP_force_plot.png", caption='SHAP Feature Importance (Waterfall)', use_container_width=True)

    st.markdown("⚙️ **Interpretation:** Positive values increase predicted concentration; negative values decrease it.")
except Exception as e:
    st.error(f"⚠️ SHAP explanation failed: {e}")

# ===============================
# 7. 教学提示
# ===============================
st.markdown("---")
st.markdown("💡 **Note:**")
st.markdown("""
- The model predicts continuous plasma concentration (ng/mL).
- ±20% denotes an empirical confidence interval.
- SHAP values indicate the magnitude and direction of feature effects.
""")


