import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. 批量加载与特征工程 (带Debug监控版)
# ==========================================
print("正在从 metadata.csv 检索全生命周期数据...")
metadata = pd.read_csv('cleaned_dataset/metadata.csv')

# 筛选出 B0005 的所有放电记录
b0005_discharge = metadata[(metadata['battery_id'] == 'B0005') & (metadata['type'] == 'discharge')]

all_cycles_data = []
cycle_count = 1

for index, row in b0005_discharge.iterrows():
    # 关键修复：加上 .strip() 强行剔除可能存在的隐藏空格
    filename = str(row['filename']).strip() 
    file_path = os.path.join('cleaned_dataset', 'data', filename)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # 安时积分法计算当前循环的 SOC
        df['dt'] = df['Time'].diff().fillna(0)
        df['Ah_consumed'] = (df['Current_measured'].abs() * df['dt']) / 3600.0
        df['Cumulative_Ah'] = df['Ah_consumed'].cumsum()
        total_capacity = df['Cumulative_Ah'].max()
        
        # 过滤掉异常的空数据
        if total_capacity > 0:
            df['SOC'] = 1 - (df['Cumulative_Ah'] / total_capacity)
            df['Cycle_Index'] = cycle_count # 记录这是第几次循环（老化特征）
            all_cycles_data.append(df)
            cycle_count += 1
        else:
            # 监控点 1：文件找到了，但数据是空的或算不出容量
            print(f"⚠️ 警告: 文件 {filename} 容量异常 (Capacity={total_capacity})，已跳过。")
    else:
        # 监控点 2：根本找不到这个文件
        print(f"❌ 报错: 找不到文件路径 {file_path}")

# 终极拦截：如果跑完循环还是空的，直接打印原因并停止
if len(all_cycles_data) == 0:
    raise RuntimeError("🚨 致命错误：没有成功加载任何有效数据！请查看上方的 ❌ 或 ⚠️ 提示来排查路径。")

# 将筛选后的表格垂直拼接成一个超大 DataFrame
full_data = pd.concat(all_cycles_data, ignore_index=True)
print(f"✅ 成功合并数据！总计提取了 {cycle_count-1} 次有效放电循环，总数据量: {len(full_data)} 行。")

# 选取输入特征：我们把"循环次数(Cycle_Index)"也加进去，让模型感知电池老化！
features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Cycle_Index']
X_raw = full_data[features].values
Y_raw = full_data['SOC'].values.reshape(-1, 1)

# 数据归一化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)

# 划分数据集：前 80% 的循环用于训练，最后 20% 的循环（电池衰减后期）用于测试
split_idx = int(len(X_scaled) * 0.8)
X_train, Y_train = X_scaled[:split_idx], Y_raw[:split_idx]
X_test, Y_test = X_scaled[split_idx:], Y_raw[split_idx:]

X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.FloatTensor(Y_train.copy())
X_test_tensor = torch.FloatTensor(X_test)
Y_test_tensor = torch.FloatTensor(Y_test.copy())

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# 数据量变大了，把批次大小（batch_size）提高，加快训练速度
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True) 

# ==========================================
# 2. 搭建模型与环境配置
# ==========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 我们把网络稍微加深一点，应对庞大的数据量
class SOCPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SOCPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SOCPredictor(input_dim=len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 3. 训练循环 (大数据量炼丹)
# ==========================================
epochs = 50 # 先跑 30 轮看看效果
print(f"\n开始在 {device} 上训练全生命周期模型...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")

# ==========================================
# 4. 测试与可视化 (抽取最后一次循环看效果)
# ==========================================
model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    Y_pred = model(X_test_device).cpu().numpy()

# ==========================================
# 5. 高阶评价与可视化 (综设高分专用)
# ==========================================
print("\n--- 正在生成多维度评估报告 ---")

# 1. 计算客观评价指标 (反归一化还原到真实的 SOC 比例)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae = mean_absolute_error(Y_test, Y_pred)
print(f"测试集均方根误差 (RMSE): {rmse:.4f}")
print(f"测试集平均绝对误差 (MAE): {mae:.4f}")

# 2. 计算每一帧的绝对误差百分比
errors = np.abs(Y_pred - Y_test) * 100 

# 创建一个 1x3 的超大画布，一次性输出三张漂亮的高级图表
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1：预测曲线局部特写 (经典的折线对比)
plot_length = min(2000, len(Y_test))
axes[0].plot(Y_test[-plot_length:], label='Real SOC', color='#2ecc71', linewidth=2)
axes[0].plot(Y_pred[-plot_length:], label='Predicted SOC', color='#e74c3c', linestyle='--', linewidth=2)
axes[0].set_title('SOC Prediction vs Ground Truth')
axes[0].set_xlabel('Time Steps')
axes[0].set_ylabel('SOC')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2：真实值与预测值的散点拟合图 (越接近对角线说明越准)
axes[1].scatter(Y_test, Y_pred, alpha=0.1, color='#3498db', s=1)
axes[1].plot([0, 1], [0, 1], color='red', linestyle='--') # 完美预测对角线
axes[1].set_title('Prediction Accuracy Scatter')
axes[1].set_xlabel('Real SOC')
axes[1].set_ylabel('Predicted SOC')
axes[1].grid(True, alpha=0.3)

# 图3：误差分布直方图 (使用 Seaborn 让图表更具高级感)
sns.histplot(errors, bins=50, kde=True, ax=axes[2], color='#9b59b6')
axes[2].set_title('Error Distribution Histogram')
axes[2].set_xlabel('Absolute Error (%)')
axes[2].set_ylabel('Frequency')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 6. 导出部署数据 (供 Streamlit 前端使用)
# ==========================================
print("\n--- 正在生成前端部署文件 ---")
# 确保数据是一维的 numpy 数组
real_soc_array = Y_test.flatten()
pred_soc_array = Y_pred.flatten()

# 整理成 DataFrame
demo_df = pd.DataFrame({
    'Time_Steps': range(len(real_soc_array)),
    'Real_SOC': real_soc_array,
    'Predicted_SOC': pred_soc_array
})

# 导出为 CSV
demo_df.to_csv('demo_results.csv', index=False)
print("✅ 前端部署数据已成功导出为 demo_results.csv！请使用 streamlit run app.py 启动网页。")