import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import re

# 读取新数据
file_path = "训练文件/训练_叉车项目202504数据.xlsx"
df = pd.read_excel(file_path)

# 只对核心字段做 dropna（允许总重量为空，防止丢失型号）
core_fields = ["锂电池型号", "电芯品牌", "电压(V)", "容量(Ah)", "尺寸(mm)"]  # 去掉 "总重量(kg)"
df = df.dropna(subset=core_fields)

# 用N/A补全所有空白字段
for col in df.columns:
    df[col] = df[col].fillna("N/A")

main_fields = [
    "锂电池型号", "电芯品牌", "电压(V)", "容量(Ah)", "尺寸(mm)", "总重量(kg)", "模组串并联方式", "适用叉车型号"
]
# 自动补齐缺失字段
for col in main_fields:
    if col not in df.columns:
        df[col] = "N/A"
df = df[main_fields + [c for c in df.columns if c not in main_fields]]

# 去除单位，仅保留数值
def extract_number(val):
    if isinstance(val, str):
        match = re.search(r"[\d.]+", val)
        return float(match.group()) if match else np.nan
    return float(val)

df["容量(Ah)"] = df["容量(Ah)"].apply(extract_number)
df["电压(V)"] = df["电压(V)"].apply(extract_number)
df["总重量(kg)"] = df["总重量(kg)"].apply(extract_number)

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
df["锂电池型号编码"] = le.fit_transform(df["锂电池型号"].astype(str))
# 电芯品牌编码
brand_le = LabelEncoder()
df["电芯品牌编码"] = brand_le.fit_transform(df["电芯品牌"].astype(str))

# 划分训练集和验证集
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("train_data.csv", index=False)
valid_df.to_csv("valid_data.csv", index=False)

# 多模型对比
from xgboost import XGBRegressor
models = {
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "GBDT": GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
}
feature_cols = ["电压(V)", "总重量(kg)", "锂电池型号编码", "电芯品牌编码"] + size_cols
target_col = "容量(Ah)"
X_train = train_df[feature_cols].fillna(0)
y_train = train_df[target_col]
X_valid = valid_df[feature_cols].fillna(0)
y_valid = valid_df[target_col]

best_r2 = -1
best_model = None
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    r2 = r2_score(y_valid, y_pred)
    print(f"{name} 验证集R²分数: {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

print(f"最优模型: {type(best_model).__name__}, R²={best_r2:.4f}")
joblib.dump(best_model, "battery_model.pkl")

# 额外导出所有非N/A的适用叉车型号全集合，供前端自动补全
try:
    df_xlsx = pd.read_excel("训练文件/训练_叉车项目202504数据.xlsx")
    all_models = [str(v).strip() for v in df_xlsx["适用叉车型号"].dropna() if str(v).strip() and str(v).strip() != "N/A"]
    with open("all_forklift_models.txt", "w", encoding="utf-8") as f:
        for m in all_models:
            f.write(m + "\n")
    print(f"已从训练文件提取所有型号到 all_forklift_models.txt，总数：{len(all_models)}")
except Exception as e:
    print(f"[WARN] all_forklift_models.txt 生成失败: {e}")

# 合并 train_data.csv 和 valid_data.csv 去重，生成 all_data.csv
try:
    train_df = pd.read_csv("train_data.csv")
    valid_df = pd.read_csv("valid_data.csv")
    all_df = pd.concat([train_df, valid_df], ignore_index=True)
    # 以“锂电池型号+适用叉车型号+电芯品牌”为唯一键去重，防止重复
    all_df = all_df.drop_duplicates(subset=["锂电池型号", "适用叉车型号", "电芯品牌"])
    all_df.to_csv("all_data.csv", index=False)
    print("已生成 all_data.csv，合并推荐数据源。")
except Exception as e:
    print(f"[WARN] all_data.csv 生成失败: {e}")
