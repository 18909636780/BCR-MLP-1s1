# 替换原有的SHAP力图代码部分
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

    # SHAP Explanation - 修改为瀑布图
    st.subheader("SHAP Waterfall Plot Explanation")
    
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
        "Tumor_Grade": "Grade",
        "Lymph_Node_Metastasis": "LN Metastasis",
        "Numbe_of_Lymph_Nodes": "LN Count",
        "Marital_Status_Unmarried": "Unmarried",
        "Marital_Status_Married": "Married",
        "Marital_Status_Divorced": "Divorced"
    }
    
    # 创建瀑布图
    plt.figure(figsize=(10, 8))
    
    # 根据预测类别选择对应的SHAP值
    if predicted_class == 1:
        shap_values_to_plot = shap_values[1][0]  # 对于类别1的SHAP值
        expected_value = explainer_shap.expected_value[1]
    else:
        shap_values_to_plot = shap_values[0][0]  # 对于类别0的SHAP值
        expected_value = explainer_shap.expected_value[0]
    
    # 创建瀑布图
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_to_plot,
            base_values=expected_value,
            data=final_features_df.iloc[0],
            feature_names=[short_feature_names.get(f, f) for f in feature_names]
        ),
        max_display=10,  # 限制显示的特征数量
        show=False
    )
    
    # 调整图形布局
    plt.tight_layout()
    plt.gcf().set_size_inches(10, 8)
    
    # 保存并显示图形
    plt.savefig("shap_waterfall.png", bbox_inches='tight', dpi=300)
    st.image("shap_waterfall.png", caption='SHAP Waterfall Plot Explanation')
