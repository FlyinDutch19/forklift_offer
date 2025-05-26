import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# 常见电芯容量
CELL_CAPACITIES = [20, 50, 65, 100, 104, 105, 150, 160, 230, 280, 304, 314]

# 读取数据
file_path = "训练_叉车项目2025.xlsx"
df = pd.read_excel(file_path)

# 数据预处理
df = df.dropna(subset=["型号", "电压(V)", "容量(Ah)", "尺寸(mm)", "总重量(kg)"])
import re

# 去除单位，仅保留数值
def extract_number(val):
    if isinstance(val, str):
        match = re.search(r"[\d.]+", val)
        return float(match.group()) if match else np.nan
    return float(val)

df["容量(Ah)"] = df["容量(Ah)"].apply(extract_number)
df["电压(V)"] = df["电压(V)"].apply(extract_number)
df["总重量(kg)"] = df["总重量(kg)"].apply(extract_number)
df["配重(kg)"] = df["配重(kg)"].fillna(0).apply(extract_number)

# 划分训练集和验证集
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存划分结果
train_df.to_csv("train_data.csv", index=False)
valid_df.to_csv("valid_data.csv", index=False)

# 保存字段名
with open("fields.txt", "w", encoding="utf-8") as f:
    f.write(",".join(df.columns.tolist()))

print("数据集划分完成，训练集和验证集已保存。")
