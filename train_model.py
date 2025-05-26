import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import re

# 常见电芯容量
CELL_CAPACITIES = [20, 50, 65, 100, 104, 105, 150, 160, 230, 280, 304, 314]

# 读取数据
file_path = "训练_叉车项目2025.xlsx"
df = pd.read_excel(file_path)

# 数据预处理
df = df.dropna(subset=["型号", "电压(V)", "容量(Ah)", "尺寸(mm)", "总重量(kg)"])

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

# 拆分尺寸为长宽高
size_cols = ["长(mm)", "宽(mm)", "高(mm)"]
def split_size(s):
    if isinstance(s, str):
        parts = re.findall(r"[\d.]+", s)
        if len(parts) == 3:
            return [float(x) for x in parts]
    return [np.nan, np.nan, np.nan]
df[size_cols] = df["尺寸(mm)"].apply(lambda x: pd.Series(split_size(x)))

# 型号编码
le = LabelEncoder()
df["型号编码"] = le.fit_transform(df["型号"].astype(str))

# 划分训练集和验证集
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("train_data.csv", index=False)
valid_df.to_csv("valid_data.csv", index=False)

# XGBoost回归
try:
    from xgboost import XGBRegressor
except ImportError:
    import os
    os.system('pip install xgboost')
    from xgboost import XGBRegressor

feature_cols = ["电压(V)", "总重量(kg)", "型号编码"] + size_cols
target_col = "容量(Ah)"

X_train = train_df[feature_cols].fillna(0)
y_train = train_df[target_col]
X_valid = valid_df[feature_cols].fillna(0)
y_valid = valid_df[target_col]

model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
r2 = r2_score(y_valid, y_pred)
print(f"模型在验证集上的准备率（R²分数）为：{r2:.4f}")

# 如需保存模型，可取消下行注释
# joblib.dump(model, "battery_model.pkl")
