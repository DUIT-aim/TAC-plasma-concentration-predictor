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
continuous_columns = ['Total_daily_dose','CL_F','BUN','BMI','ALB','NE','CCR','IBIL','Dosing_time']  #分类变量
#columns_to_copy = ['CYP3A5']  # 分类变量

# 在 Streamlit 界面上创建输入框
st.sidebar.header("Please enter the patient's details")

Total_daily_dose = st.sidebar.number_input("Total daily dose (mg):", min_value=0.5, max_value=10.0, value=5.0)
CL_F = st.sidebar.number_input("CL/F (L/h):", min_value=15.0, max_value=30.0, value=22.5)
BUN = st.sidebar.number_input("BUN (mmol/L):", min_value=2.0, max_value=40.0, value=11.5)
BMI = st.sidebar.number_input("BMI (kg/m²):", min_value=15.0, max_value=40.0, value=24.5)
ALB = st.sidebar.number_input("ALB (g/L):", min_value=10.0, max_value=60.0, value=35.0)
NE = st.sidebar.number_input("NE# (10⁹/L):", min_value=0.5, max_value=25.0, value=6.5)
CCR = st.sidebar.number_input("CCR (mL/min):", min_value=15.0, max_value=350.0, value=115.0)
IBIL = st.sidebar.number_input("IBIL (µmol/L):", min_value=0.0, max_value=10.0, value=5.0)
Dosing time = st.sidebar.number_input("Dosing time (day):", min_value=0.0, max_value=500.0, value=200.0)

# 汇总输入
input_data = np.array([[Total_daily_dose, CL_F, BUN, BMI, ALB, NE, CCR, IBIL,Dosing_time]])

# 转换为 DataFrame，便于后续标准化与 SHAP 解释
input_df = pd.DataFrame(input_data, columns=continuous_columns)

# ===============================
# 4. 标准化输入
# ===============================
input_scaled = scaler.transform(input_df)

# ===============================
# 5. 模型预测
# ===============================
if st.button("Predict Tacrolimus Plasma Concentration"):
    # 预测连续值
    predicted_value = model.predict(input_scaled)[0]

    # 计算 ±20% 区间
    lower_bound = predicted_value * 0.8
    upper_bound = predicted_value * 1.2

    # 输出预测结果
    st.subheader("🧪 Predicted Result")
    st.write(f"**Tacrolimus Plasma Concentration = {predicted_value:.2f} ± 20% ng/mL**")
    st.write(f"Estimated range: {lower_bound:.2f} – {upper_bound:.2f} ng/mL")

# ===============================
# 6. SHAP 力图解释（带空格/符号的特征名）
# ===============================
st.subheader("🔍 SHAP Feature Importance Explanation")

try:
    # 使用训练数据的小样本作为背景
    df_train = pd.read_csv('train.csv', encoding='utf-8')
    X_train = df_train[continuous_columns]
    X_train_scaled = scaler.transform(X_train)

    # 设置模型输入列名与展示列名
    continuous_columns = ['Total_daily_dose','CL_F','BUN','BMI','ALB','NE','CCR','IBIL','Dosing_time']
    display_names = ['Total daily dose','CL/F','BUN','BMI','ALB','NE#','CCR','IBIL','Dosing time']

    # 构建 DataFrame 并替换列名为展示名
    X_train_scaled_df = pd.DataFrame(X_train_scaled[:50], columns=display_names)
    input_scaled_df = pd.DataFrame(input_scaled, columns=display_names)

    # 建立解释器
    explainer = shap.Explainer(model.predict, X_train_scaled_df)
    shap_values = explainer(input_scaled_df)

    # 绘制 waterfall 图
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("SHAP_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("SHAP_force_plot.png", caption='SHAP Feature Importance (Waterfall)', use_container_width=True)

    st.markdown("⚙️ **Interpretation:** Positive values increase the predicted concentration; negative values decrease it.")
except Exception as e:
    st.error(f"⚠️ SHAP explanation failed: {e}")

# ===============================
# 7. 教学提示
# ===============================
st.markdown("---")
st.markdown("💡 **Attention please：**")
st.markdown("""
-This model is a continuous prediction, outputting plasma concentration (ng/mL).
- '±20%' denotes an empirical confidence interval, within which actual plasma drug concentrations are considered reasonable.
- SHAP values can be used to observe the direction and magnitude of the influence of features on individual predictions.。
""")