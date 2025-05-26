import pandas as pd

# 读取Excel文件
file_path = "训练_叉车项目2025.xlsx"
df = pd.read_excel(file_path)

# 打印字段名和前5行数据
print("字段名：")
print(df.columns.tolist())
print("\n前5行数据：")
print(df.head())
