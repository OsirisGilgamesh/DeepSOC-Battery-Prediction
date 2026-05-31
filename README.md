# 🔋 DeepSOC-Battery-Prediction

> 基于特征感知深度学习的锂离子电池 SOC 实时预测与在线训练边缘评估系统。

本仓库包含了《人工智能选修课》期末实践项目的完整代码与部署方案。项目旨在解决锂离子电池在全生命周期老化后期，由于内部复杂的电化学非线性变化导致荷电状态（State of Charge, SOC）难以精准预测的工程痛点。

系统采用前馈神经网络多层感知机（MLP）作为核心算法，并在 PyTorch 框架下通过 Apple Silicon MPS 实现硬件加速，最终通过 Streamlit 实现了端侧的可视化部署。

---

## ✨ 核心特性 (Key Features)

- **特征感知工程**：除常规的电压(V)、电流(I)、温度(T)外，创新性引入“循环次数(Cycle_Index)”作为老化先验特征，赋予黑盒模型物理感知能力。
- **高精度极化跟踪**：在完全不依赖历史时序序列（如 LSTM）的前提下，对 NASA B0005 电池寿命末期（最后20%重度衰减期）的极限盲测中，取得了 **MAE 1.36%**，**RMSE 0.0167** 的优异成绩。
- **端侧系统部署**：剥离笨重的训练流，使用 Streamlit + Plotly 搭建了极具工程落地感的在线 BMS（电池管理系统）前端。
- **云端微调模拟**：系统不仅支持内置预训练模型的实时推断，还提供自定义老化数据集上传与动态微调训练功能。

---

## 🛠️ 技术栈 (Tech Stack)

- **核心语言**：Python 3.9+
- **深度学习框架**：PyTorch 2.0+ (支持 Apple M-Series MPS 硬件加速)
- **数据科学库**：Pandas, NumPy, Scikit-learn (MinMaxScaler)
- **前端部署与可视化**：Streamlit, Plotly, Matplotlib, Seaborn

---

## 📂 项目结构 (Project Structure)

```text
DeepSOC-Battery-Prediction/
├── train_full_soc.py      # 核心训练脚本：包含数据加载、MLP构建、MPS加速训练与图表评估
├── app.py                 # 前端部署脚本：Streamlit 边缘计算大屏与在线训练模拟器
├── demo_results.csv       # 模型部署数据：由 train_full_soc.py 导出，供前端大屏调取
├── test_battery_data.csv  # 测试数据：用于演示平台 Tab 2 的自定义新电池上传与微调功能
└── README.md              # 项目说明文档
