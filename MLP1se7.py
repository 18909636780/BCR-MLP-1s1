###Streamlit应用程序开发
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


# Define feature names
feature_names = ["Tumor_Size_at_Diagnosis", "Tumor_Grade", "Lymph_Node_Metastasis", "Numbe_of_Lymph_Nodes", "Marital_Status_Unmarried", "Marital_Status_Married", "Marital_Status_Divorced"]

# Streamlit user interface
st.markdown("<h3 style='text-align: left;'>Breast Cancer Recurrence Predictor</h3>", unsafe_allow_html=True)

# Tumor_Size_at_Diagnosis
Tumor_Size_at_Diagnosis = st.number_input("Tumor Size at Diagnosis(mm):", min_value=0, max_value=100, value=25)

# Tumor_Grade
Tumor_Grade = st.selectbox("Tumor Grade:", options=list(Tumor_Grade_options.keys()), format_func=lambda x: Tumor_Grade_options[x])

# Lymph_Node_Metastasis
Lymph_Node_Metastasis = st.selectbox("Lymph Node Metastasis:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Numbe_of_Lymph_Nodes
Numbe_of_Lymph_Nodes = st.number_input("Numbe of Lymph Nodes:", min_value=0, max_value=50, value=0)

# Marital_Status_Unmarried
Marital_Status_Unmarried = st.selectbox("Marital Status Unmarried:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Marital_Status_Married
Marital_Status_Married = st.selectbox("Marital Status Married:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Marital_Status_Divorced
Marital_Status_Divorced = st.selectbox("Marital Status Divorced:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')


# 准备输入特征
feature_values = [Tumor_Size_at_Diagnosis, Tumor_Grade, Lymph_Node_Metastasis, Numbe_of_Lymph_Nodes, Marital_Status_Unmarried, Marital_Status_Married, Marital_Status_Divorced]
features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [Tumor_Size_at_Diagnosis, Numbe_of_Lymph_Nodes]
categorical_features=[Tumor_Grade,Lymph_Node_Metastasis,Marital_Status_Unmarried,Marital_Status_Married,Marital_Status_Divorced]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)


# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=["Tumor_Size_at_Diagnosis", "Numbe_of_Lymph_Nodes"])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)


# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=feature_names)


if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(final_features_df)[0]   
    predicted_proba = model.predict_proba(final_features_df)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}(0: No Disease,1: Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
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

    # 创建SHAP解释器
    df = pd.read_csv('modified_train_data_1se3.csv', encoding='utf8')
    ytrain = df.Recurrence_after_2_Years
    x_train = df.drop('Recurrence_after_2_Years', axis=1)
    
    continuous_cols = [1,4]
    xtrain = x_train.copy()
    scaler = StandardScaler()
    xtrain.iloc[:, continuous_cols] = scaler.fit_transform(x_train.iloc[:, continuous_cols])

    explainer_shap = shap.KernelExplainer(model.predict_proba, xtrain)
    
    # 获取SHAP值
    shap_values = explainer_shap.shap_values(final_features_df)
    
    # 准备绘图
    plt.figure(figsize=(12, 4), dpi=120)
    
    # 使用HTML方式显示force plot
    if predicted_class == 1:
        force_plot = shap.force_plot(
            explainer_shap.expected_value[1],
            shap_values[1][0],  # 取第一个样本的SHAP值
            final_features_df.iloc[0],
            feature_names=feature_names,
            show=False,
            matplotlib=False  # 关键修改：不使用matplotlib
        )
    else:
        force_plot = shap.force_plot(
            explainer_shap.expected_value[0],
            shap_values[0][0],  # 取第一个样本的SHAP值
            final_features_df.iloc[0],
            feature_names=feature_names,
            show=False,
            matplotlib=False  # 关键修改：不使用matplotlib
        )
    
    # 保存为HTML文件然后显示
    shap.save_html("shap_force_plot.html", force_plot)
    with open("shap_force_plot.html", "r") as f:
        html = f.read()
    st.components.v1.html(html, height=400, scrolling=True)

    # 版本兼容性警告处理
    st.warning("""
    Note: You're seeing this because of version differences between the saved model and current scikit-learn. 
    This shouldn't affect predictions but consider retraining your model with the current version for best compatibility.
    """)
