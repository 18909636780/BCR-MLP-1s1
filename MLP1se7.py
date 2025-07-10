import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载模型
model = joblib.load('MLP1se7.pkl')
scaler = joblib.load('scaler1se7.pkl') 

# 定义特征选项
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

# 定义特征名称
feature_names = ["Tumor_Size_at_Diagnosis", "Tumor_Grade", "Lymph_Node_Metastasis", 
                "Numbe_of_Lymph_Nodes", "Marital_Status_Unmarried", 
                "Marital_Status_Married", "Marital_Status_Divorced"]

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    # 左侧列 - 输入变量
    st.title("Breast Cancer Recurrence Predictor")
    st.subheader("Input Parameters")
    
    # Tumor_Size_at_Diagnosis
    Tumor_Size_at_Diagnosis = st.number_input("Tumor Size at Diagnosis(mm):", min_value=0, max_value=100, value=25)

    # Tumor_Grade
    Tumor_Grade = st.selectbox("Tumor Grade:", options=list(Tumor_Grade_options.keys()), 
                             format_func=lambda x: Tumor_Grade_options[x])

    # Lymph_Node_Metastasis
    Lymph_Node_Metastasis = st.selectbox("Lymph Node Metastasis:", options=[0, 1], 
                                       format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

    # Numbe_of_Lymph_Nodes
    Numbe_of_Lymph_Nodes = st.number_input("Number of Lymph Nodes:", min_value=0, max_value=50, value=0)

    # Marital_Status_Unmarried
    Marital_Status_Unmarried = st.selectbox("Marital Status Unmarried:", options=[0, 1], 
                                          format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

    # Marital_Status_Married
    Marital_Status_Married = st.selectbox("Marital Status Married:", options=[0, 1], 
                                        format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

    # Marital_Status_Divorced
    Marital_Status_Divorced = st.selectbox("Marital Status Divorced:", options=[0, 1], 
                                         format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

    # 准备输入特征
    feature_values = [Tumor_Size_at_Diagnosis, Tumor_Grade, Lymph_Node_Metastasis, 
                     Numbe_of_Lymph_Nodes, Marital_Status_Unmarried, 
                     Marital_Status_Married, Marital_Status_Divorced]
    features = np.array([feature_values])

    # 分离连续变量和分类变量
    continuous_features = [Tumor_Size_at_Diagnosis, Numbe_of_Lymph_Nodes]
    categorical_features = [Tumor_Grade, Lymph_Node_Metastasis, Marital_Status_Unmarried, 
                          Marital_Status_Married, Marital_Status_Divorced]

    # 对连续变量进行标准化
    continuous_features_array = np.array(continuous_features).reshape(1, -1)
    continuous_features_df = pd.DataFrame(continuous_features_array, 
                                        columns=["Tumor_Size_at_Diagnosis", "Numbe_of_Lymph_Nodes"])
    continuous_features_standardized = scaler.transform(continuous_features_df)

    # 将标准化后的连续变量和原始分类变量合并
    categorical_features_array = np.array(categorical_features).reshape(1, -1)
    final_features = np.hstack([continuous_features_standardized, categorical_features_array])
    final_features_df = pd.DataFrame(final_features, columns=feature_names)

# 预测按钮
if st.button("Predict", key="predict_button"):
    with col2:
        # 右侧列 - 预测结果和解释
        st.title("Prediction Results")
        
        # 预测类别和概率
        predicted_class = model.predict(final_features_df)[0]   
        predicted_proba = model.predict_proba(final_features_df)[0]

        # 显示预测结果    
        st.write(f"**Predicted Class:** {'Disease (1)' if predicted_class == 1 else 'No Disease (0)'}")   
        st.write(f"**Probability of Disease:** {predicted_proba[1]:.2%}")
        st.write(f"**Probability of No Disease:** {predicted_proba[0]:.2%}")

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100
        if predicted_class == 1:        
            advice = (            
                f"🔴 **High Risk Alert:** The model predicts a {probability:.1f}% probability of breast cancer recurrence. "
                "It's strongly advised to consult with your healthcare provider for further evaluation "
                "and possible intervention."        
            )    
        else:        
            advice = (           
                f"🟢 **Low Risk:** The model predicts a {probability:.1f}% probability of no recurrence. "
                "Maintain regular check-ups and a healthy lifestyle as preventive measures."        
            )    
        st.success(advice)

        # SHAP瀑布图解释
        st.subheader("Feature Impact Analysis (SHAP)")
        
        # 创建SHAP解释器
        df = pd.read_csv('modified_train_data_1se3.csv', encoding='utf8')
        ytrain = df.Recurrence_after_2_Years
        x_train = df.drop('Recurrence_after_2_Years', axis=1)
        continuous_cols = [1, 4]
        xtrain = x_train.copy()
        scaler = StandardScaler()
        xtrain.iloc[:, continuous_cols] = scaler.fit_transform(x_train.iloc[:, continuous_cols])

        explainer_shap = shap.KernelExplainer(model.predict_proba, xtrain)
        
        # 获取SHAP值
        shap_values = explainer_shap.shap_values(final_features_df)
        
        # 创建简短的特征名称用于显示
        short_feature_names = {
            "Tumor_Size_at_Diagnosis": "Tumor Size",
            "Tumor_Grade": "Tumor Grade",
            "Lymph_Node_Metastasis": "LN Metastasis",
            "Numbe_of_Lymph_Nodes": "LN Count",
            "Marital_Status_Unmarried": "Unmarried",
            "Marital_Status_Married": "Married",
            "Marital_Status_Divorced": "Divorced"
        }
        
        # 创建瀑布图
        plt.figure(figsize=(10, 6))
        
        # 根据预测类别选择对应的SHAP值
        if predicted_class == 1:
            shap_values_to_plot = shap_values[1][0]  # 对于类别1的SHAP值
            expected_value = explainer_shap.expected_value[1]
            title = "Factors Increasing Recurrence Risk"
        else:
            shap_values_to_plot = shap_values[0][0]  # 对于类别0的SHAP值
            expected_value = explainer_shap.expected_value[0]
            title = "Factors Decreasing Recurrence Risk"
        
        # 创建瀑布图
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_to_plot,
                base_values=expected_value,
                data=final_features_df.iloc[0],
                feature_names=[short_feature_names.get(f, f) for f in feature_names]
            ),
            max_display=7,  # 显示所有7个特征
            show=False
        )
        
        # 调整图形布局
        plt.title(title, fontsize=12)
        plt.tight_layout()
        
        # 在Streamlit中显示图形
        st.pyplot(plt.gcf())
        
        # 添加图例说明
        st.caption("""
        **How to interpret this chart:**
        - Bars to the right (positive values) indicate factors increasing recurrence risk
        - Bars to the left (negative values) indicate factors decreasing recurrence risk
        - The length of each bar shows the magnitude of the feature's impact
        """)
