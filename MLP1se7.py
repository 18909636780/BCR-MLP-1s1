### Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('MLP1se7.pkl')
scaler = joblib.load('scaler1se7.pkl') 

# Define feature options
Level_of_Education_options = {    
    0: 'Primary(0)',    
    1: 'Secondary(1)',    
    2: 'Certificate(2)',    
    3: 'Diploma(3)',
    4: 'Degree(4)'
}

Tumor_Grade_options = {       
    1: 'Grade1',    
    2: 'Grade2',    
    3: 'Grade3',
    4: 'Grade4'
}

# Define feature names (使用更简洁的显示名称)
display_feature_names = {
    "Tumor_Size_at_Diagnosis": "Tumor Size",
    "Tumor_Grade": "Tumor Grade",
    "Lymph_Node_Metastasis": "Lymph Node Meta",
    "Numbe_of_Lymph_Nodes": "Lymph Nodes Count",
    "Marital_Status_Unmarried": "Unmarried",
    "Marital_Status_Married": "Married",
    "Marital_Status_Divorced": "Divorced"
}

# Original feature names for model input
feature_names = ["Tumor_Size_at_Diagnosis", "Tumor_Grade", "Lymph_Node_Metastasis", 
                 "Numbe_of_Lymph_Nodes", "Marital_Status_Unmarried", 
                 "Marital_Status_Married", "Marital_Status_Divorced"]

# Streamlit user interface
st.title("Breast Cancer Recurrence Predictor")

# 输入控件保持不变...
Tumor_Size_at_Diagnosis = st.number_input("Tumor Size at Diagnosis(mm):", min_value=0, max_value=100, value=25)
Tumor_Grade = st.selectbox("Tumor Grade:", options=list(Tumor_Grade_options.keys()), format_func=lambda x: Tumor_Grade_options[x])
Lymph_Node_Metastasis = st.selectbox("Lymph Node Metastasis:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Numbe_of_Lymph_Nodes = st.number_input("Numbe of Lymph Nodes:", min_value=0, max_value=50, value=0)
Marital_Status_Unmarried = st.selectbox("Marital Status Unmarried:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Marital_Status_Married = st.selectbox("Marital Status Married:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Marital_Status_Divorced = st.selectbox("Marital Status Divorced:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# 准备输入特征
feature_values = [Tumor_Size_at_Diagnosis, Tumor_Grade, Lymph_Node_Metastasis, 
                  Numbe_of_Lymph_Nodes, Marital_Status_Unmarried, 
                  Marital_Status_Married, Marital_Status_Divorced]
features = np.array([feature_values])

# 数据处理保持不变...
continuous_features = [Tumor_Size_at_Diagnosis, Numbe_of_Lymph_Nodes]
categorical_features = [Tumor_Grade, Lymph_Node_Metastasis, Marital_Status_Unmarried, 
                       Marital_Status_Married, Marital_Status_Divorced]

continuous_features_array = np.array(continuous_features).reshape(1, -1)
continuous_features_df = pd.DataFrame(continuous_features_array, 
                                     columns=["Tumor_Size_at_Diagnosis", "Numbe_of_Lymph_Nodes"])
continuous_features_standardized = scaler.transform(continuous_features_df)

categorical_features_array = np.array(categorical_features).reshape(1, -1)
final_features = np.hstack([continuous_features_standardized, categorical_features_array])
final_features_df = pd.DataFrame(final_features, columns=feature_names)

if st.button("Predict"):
    # 预测部分保持不变...
    predicted_class = model.predict(final_features_df)[0]
    predicted_proba = model.predict_proba(final_features_df)[0]
    
    st.write(f"**Predicted Class:** {predicted_class}(0: No Disease,1: Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of breast cancer recurrence. "
            f"The model predicts that your probability of having breast cancer recurrence is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of breast cancer recurrence. "
            f"The model predicts that your probability of not having breast cancer recurrence is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    
    # 加载训练数据
    df = pd.read_csv('modified_train_data_1se3.csv', encoding='utf8')
    ytrain = df.Recurrence_after_2_Years
    x_train = df.drop('Recurrence_after_2_Years', axis=1)
    
    # 标准化训练数据
    continuous_cols = [1, 4]
    xtrain = x_train.copy()
    scaler = StandardScaler()
    xtrain.iloc[:, continuous_cols] = scaler.fit_transform(x_train.iloc[:, continuous_cols])
    
    # 创建SHAP解释器
    explainer_shap = shap.KernelExplainer(model.predict_proba, xtrain)
    
    # 获取SHAP值
    shap_values = explainer_shap.shap_values(final_features_df)
    
    # 使用原始特征值（标准化前的值）
    original_feature_values = pd.DataFrame(features, columns=feature_names)
    
    # 设置图形大小和字体
    plt.rcParams.update({'font.size': 8})  # 调小字体大小
    plt.figure(figsize=(10, 4))  # 设置更大的图形尺寸
    
    # 创建SHAP force plot
    if predicted_class == 1:
        shap.force_plot(
            explainer_shap.expected_value[1], 
            shap_values[1][0],  # 使用[1][0]获取第一个样本的SHAP值
            original_feature_values.iloc[0], 
            matplotlib=True,
            feature_names=display_feature_names,  # 使用更简洁的显示名称
            plot_cmap=["#77dd77", "#f99191"],  # 自定义颜色
            text_rotation=15,  # 轻微旋转文本
            show=False  # 不立即显示，以便我们可以调整
        )
    else:
        shap.force_plot(
            explainer_shap.expected_value[0], 
            shap_values[0][0],  # 使用[0][0]获取第一个样本的SHAP值
            original_feature_values.iloc[0], 
            matplotlib=True,
            feature_names=display_feature_names,  # 使用更简洁的显示名称
            plot_cmap=["#77dd77", "#f99191"],  # 自定义颜色
            text_rotation=15,  # 轻微旋转文本
            show=False  # 不立即显示，以便我们可以调整
        )
    
    # 调整布局并保存图像
    plt.tight_layout()  # 自动调整布局防止重叠
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=150, pad_inches=0.5)
    plt.close()
    
    # 显示图像
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
