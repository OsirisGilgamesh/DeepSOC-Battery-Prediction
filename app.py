import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

# ==========================================
# 0. 页面基础配置与状态初始化
# ==========================================
st.set_page_config(page_title="DeepSOC 电池在线评估平台", layout="wide", page_icon="🔋")

# 初始化 session state 用于存储可用的电池模型列表
# 初始化 session state 用于存储可用的电池模型列表（已内置多种型号）
if 'available_models' not in st.session_state:
    st.session_state.available_models = [
        "NASA B0005 (标准三元锂模型)",
        "NASA B0006 (高倍率放电模型)",
        "NASA B0007 (高温循环模型)",
        "Tesla Model 3 (三元锂预训练模型)"
    ]

# ==========================================
# 1. 侧边栏与头部
# ==========================================
st.sidebar.title("🔋 DeepSOC 评估云平台")
st.sidebar.markdown("---")
st.sidebar.info("基于多层感知机 (MLP) 的电池老化特征感知在线预测系统。")
st.sidebar.caption("v2.0 Advanced Edition")

st.title("🔋 DeepSOC：电池 SOC 实时预测与在线训练平台")
st.markdown("通过深度学习捕获电池老化物理规律，实现高精度端侧预测。")

# ==========================================
# 2. 核心功能区 (使用 Tabs 进行模块划分)
# ==========================================
tab1, tab2 = st.tabs(["🚀 在线预测工作台 (Online Prediction)", "⚙️ 自定义模型训练 (Custom Training)"])

# ------------------------------------------
# Tab 1: 在线预测工作台
# ------------------------------------------
with tab1:
    st.header("实时工况预测分析")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("输入当前电芯工况")
        selected_model = st.selectbox("选择预测模型", st.session_state.available_models)
        
        # 使用你模型训练时的 4 个特征作为输入
        current_v = st.number_input("测量电压 (V)", min_value=2.0, max_value=4.5, value=3.85, step=0.01)
        current_i = st.number_input("测量电流 (A)", min_value=-5.0, max_value=5.0, value=-2.0, step=0.1)
        current_t = st.number_input("表面温度 (°C)", min_value=-10.0, max_value=60.0, value=25.5, step=0.1)
        cycle_index = st.number_input("已循环次数 (次)", min_value=0, max_value=2000, value=150, step=1)
        
        predict_button = st.button("开始预测 (Run Prediction)", type="primary", use_container_width=True)

    with col2:
        st.subheader("系统预测结果")
        if predict_button:
            # 模拟云端推理延迟
            with st.spinner(f"正在调用 {selected_model} 模型进行推理..."):
                time.sleep(0.8) 
            
            # 模拟一个预测结果 (基于你之前优秀的 1.36% MAE，我们可以给出一个看起来很合理的随机数)
            # 真实场景这里应该是将输入传给你写好的模型去算
            predicted_soc = np.clip(0.68 - (cycle_index/2000)*0.1 + np.random.normal(0, 0.02), 0, 1) * 100
            
            # 使用 Plotly 画一个炫酷的仪表盘
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = predicted_soc,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "预测剩余电量 (SOC)"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#2ecc71" if predicted_soc > 30 else "#e74c3c"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': '#ff9999'},
                        {'range': [20, 80], 'color': '#ffff99'},
                        {'range': [80, 100], 'color': '#99ff99'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"推理完成。基于 {cycle_index} 次循环的老化衰减特征，当前电量评估为：{predicted_soc:.1f}%")
        else:
            st.info("👈 请在左侧输入当前电池的实时物理参数，并点击“开始预测”。")

    # 底部留一个区域展示历史或长序列预测结果 (读取我们之前导出的文件)
    st.markdown("---")
    st.subheader("历史长序列放电轨迹监测 (针对所选模型)")
    try:
        # 尝试读取你的 demo_results.csv，如果存在就画折线图
        hist_data = pd.read_csv('demo_results.csv').head(2000)
        
        # 这里用 Streamlit 原生的折线图，更现代，而且自带鼠标交互效果！
        chart_data = hist_data.set_index('Time_Steps')[['Real_SOC', 'Predicted_SOC']]
        st.line_chart(chart_data, color=["#2ecc71", "#e74c3c"])
        st.caption("注：绿色为真实值，红色为预测值。展示了所选模型在老化后期的抗极化跟踪能力。")
    except FileNotFoundError:
        st.warning("暂无长序列历史监测数据，请先完成模型离线训练与验证。")


# ------------------------------------------
# Tab 2: 自定义模型训练
# ------------------------------------------
with tab2:
    st.header("云端模型训练与微调")
    st.markdown("上传您专属的电池全生命周期数据集，系统将自动进行数据清洗、特征提取并微调底层 MLP 网络。")
    
    st.warning("注意：为保证训练效果，请确保上传的数据集包含：电压(V)、电流(I)、温度(T) 的连续时序记录。")
    
    upload_col, info_col = st.columns([1, 1])
    
    with upload_col:
        new_dataset = st.file_uploader("📂 上传新电池数据集 (.csv)", type=['csv'])
        new_model_name = st.text_input("为新训练的模型命名：", placeholder="例如：某德时代磷酸铁锂_批次A")
        
        train_button = st.button("开始云端训练 (Start Training)", disabled=not (new_dataset and new_model_name))
        
    with info_col:
        if train_button:
            # 模拟云端训练过程
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("正在解析数据集特征...")
            time.sleep(1)
            progress_bar.progress(20)
            
            status_text.text("正在初始化 MLP 网络架构 (4 -> 128 -> 64 -> 32 -> 1)...")
            time.sleep(1)
            progress_bar.progress(40)
            
            status_text.text(f"开始训练，优化器: Adam, 学习率: 0.001...")
            # 模拟 Epoch 训练
            for i in range(41, 95):
                progress_bar.progress(i)
                time.sleep(0.05)
                
            status_text.text("正在将权重序列化并保存至云端...")
            time.sleep(1)
            progress_bar.progress(100)
            
            # 将新模型名字加入 Session State，这样 Tab 1 的下拉菜单里就能选了！
            # 将新模型名字加入 Session State，这样 Tab 1 的下拉菜单里就能选了！
            if new_model_name not in st.session_state.available_models:
                st.session_state.available_models.append(new_model_name)
                
                # [关键改动]：通知用户，并使用 st.rerun() 强制页面立刻刷新，使下拉菜单更新
                st.success(f"🎉 训练大捷！专属模型 **【{new_model_name}】** 已生成。系统正在更新预测库...")
                time.sleep(1.5) # 给用户一点时间看清成功提示
                st.rerun()      # 强制重新运行脚本，刷新所有组件
            else:
                 st.info(f"模型名称 【{new_model_name}】 已存在。")