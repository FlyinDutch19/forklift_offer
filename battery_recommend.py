# battery_recommend.py
# 业务逻辑与数据处理
import pandas as pd
import numpy as np

train_df = pd.read_csv("train_data.csv")
df = train_df  # 保持后续兼容
VOLTAGE_MAP = dict(zip(train_df["电压(V)"], train_df["对应铅酸电池电压(V)"]))
CELL_CAPACITIES = sorted(set(train_df["单体电芯容量(Ah)"].dropna().astype(int)))

EUR_USD_RATE = 1.09

def recommend_battery(input_data):
    # 解析原电池尺寸
    input_size = input_data.get("原电池尺寸(mm)", "").replace("×", "*").replace("x", "*").replace("X", "*")
    input_size_tuple = None
    if input_size and "*" in input_size:
        try:
            input_size_tuple = tuple(float(x) for x in input_size.split("*"))
            if len(input_size_tuple) != 3:
                input_size_tuple = None
        except Exception:
            input_size_tuple = None
    input_weight = float(input_data.get("总重量(kg)", 0) or 0)

    def size_within_limit(bat_size, limit_size):
        try:
            bat_tuple = tuple(float(x) for x in str(bat_size).replace("×", "*").replace("x", "*").replace("X", "*").split("*"))
            if len(bat_tuple) != 3 or not limit_size:
                return True
            return all(b <= l for b, l in zip(sorted(bat_tuple), sorted(limit_size)))
        except Exception:
            return True

    # 1. 叉车型号优先推荐（模糊匹配）
    if "适用叉车型号" in input_data and input_data["适用叉车型号"]:
        match = train_df[train_df["适用叉车型号"] == input_data["适用叉车型号"]]
        if not match.empty:
            result = match.iloc[0].to_dict()
            # 补全所有推荐字段
            if not result.get("锂电池型号"):
                for alt in ["推荐电池型号", "型号", "电池型号"]:
                    if result.get(alt):
                        result["锂电池型号"] = result[alt]
                        break
            # 补全所有字段
            for field in [
                "型号", "锂电池型号", "电压(V)", "对应铅酸电池电压(V)", "容量(Ah)", "单体电芯容量(Ah)", "尺寸(mm)", "总重量(kg)", "配重(kg)", "适用叉车型号"
            ]:
                if field not in result:
                    result[field] = "-"
        else:
            fuzzy = train_df[train_df["适用叉车型号"].astype(str).str.contains(str(input_data["适用叉车型号"]), case=False, na=False)]
            if not fuzzy.empty:
                result = fuzzy.iloc[0].to_dict()
                # 补全所有推荐字段
                if not result.get("锂电池型号"):
                    for alt in ["推荐电池型号", "型号", "电池型号"]:
                        if result.get(alt):
                            result["锂电池型号"] = result[alt]
                            break
                for field in [
                    "型号", "锂电池型号", "电压(V)", "对应铅酸电池电压(V)", "容量(Ah)", "单体电芯容量(Ah)", "尺寸(mm)", "总重量(kg)", "配重(kg)", "适用叉车型号"
                ]:
                    if field not in result:
                        result[field] = "-"
            else:
                # 生成新型号
                raw_capacity = float(input_data.get("容量(Ah)", 0))
                cell_capacity = min(CELL_CAPACITIES, key=lambda x: abs(x - raw_capacity))
                n_cells = int(np.ceil(raw_capacity / cell_capacity))
                final_capacity = n_cells * cell_capacity
                raw_voltage = float(input_data.get("电压(V)", 0))
                std_voltage = min(VOLTAGE_MAP.keys(), key=lambda v: abs(v - raw_voltage)) if raw_voltage else 51.2
                code_voltage = int(VOLTAGE_MAP[std_voltage])
                new_model = f"F{code_voltage}{int(final_capacity)}"
                weight = float(input_data.get("总重量(kg)", 0))
                match_row = train_df[(np.isclose(train_df["电压(V)"], std_voltage, atol=1)) & (np.isclose(train_df["容量(Ah)"], final_capacity, atol=5))]
                if not match_row.empty:
                    counter_weight = float(match_row.iloc[0]["配重(kg)"])
                else:
                    counter_weight = 0.0
                # 尺寸约束：推荐尺寸不能大于原电池尺寸
                if input_size_tuple:
                    std_size = "-"
                    for s in train_df[train_df["电压(V)"] == std_voltage]["尺寸(mm)"]:
                        if size_within_limit(s, input_size_tuple):
                            std_size = s
                            break
                    else:
                        std_size = "-"
                else:
                    std_size = "-"
                # 重量补齐逻辑
                if input_weight > 0:
                    if weight < input_weight:
                        final_weight = int(round(input_weight))
                        final_counter = int(round(input_weight - weight))
                    else:
                        final_weight = int(round(weight))
                        final_counter = 0
                else:
                    final_weight = int(round(weight))
                    final_counter = 0
                hz_price = 230 * code_voltage * final_capacity / 1000 + counter_weight * 1.5
                nl_price = hz_price * 1.2 / EUR_USD_RATE
                return {
                    "适用叉车型号": input_data.get("适用叉车型号", "-"),
                    "锂电池型号": new_model,
                    "电压(V)": std_voltage,
                    "对应铅酸电池电压(V)": code_voltage,
                    "容量(Ah)": final_capacity,
                    "单体电芯容量(Ah)": cell_capacity,
                    "尺寸(mm)": std_size,
                    "总重量(kg)": final_weight,
                    "含配重(kg)": final_counter,
                    "惠州出厂价(USD)": f"{hz_price:.2f}",
                    "荷兰EXW出货价(EUR)": f"{nl_price:.2f}",
                    "汇率(USD/EUR)": f"1 EUR = {EUR_USD_RATE:.4f} USD",
                    "AI推荐": "未找到匹配结果"
                }
        code_voltage = int(result["对应铅酸电池电压(V)"])
        final_capacity = int(result["容量(Ah)"])
        counter_weight = float(result.get("配重(kg)", 0) or 0)
        hz_price = 230 * code_voltage * final_capacity / 1000 + counter_weight * 1.5
        nl_price = hz_price * 1.2 / EUR_USD_RATE
        result["惠州出厂价(USD)"] = f"{hz_price:.2f}"
        result["荷兰EXW出货价(EUR)"] = f"{nl_price:.2f}"
        result["汇率(USD/EUR)"] = f"1 EUR = {EUR_USD_RATE:.4f} USD"
        result["总重量(kg)"] = int(round(result.get("总重量(kg)", 0) or 0))
        result["含配重(kg)"] = int(round(result.get("配重(kg)", 0) or 0))
        for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)", "汇率(USD/EUR)"]:
            v = result.pop(k)
            result[k] = v
        return result
    # 2. 原电池类型与参数推荐
    if "原电池类型" in input_data and input_data["原电池类型"] == "锂电池":
        cond = (
            (np.isclose(df["电压(V)"], input_data.get("电压(V)", 0), atol=1))
            & (np.isclose(df["容量(Ah)"], input_data.get("容量(Ah)", 0), atol=5))
            & (np.isclose(df["总重量(kg)"], input_data.get("总重量(kg)", 0), atol=5))
        )
        match = df[cond]
        if not match.empty:
            result = match.iloc[0].to_dict()
            # 补全所有推荐字段
            if not result.get("锂电池型号"):
                for alt in ["推荐电池型号", "型号", "电池型号"]:
                    if result.get(alt):
                        result["锂电池型号"] = result[alt]
                        break
            for field in [
                "型号", "锂电池型号", "电压(V)", "对应铅酸电池电压(V)", "容量(Ah)", "单体电芯容量(Ah)", "尺寸(mm)", "总重量(kg)", "配重(kg)", "适用叉车型号"
            ]:
                if field not in result:
                    result[field] = "-"
            # 尺寸和重量约束过滤
            if input_size_tuple:
                if not size_within_limit(result.get("尺寸(mm)", "-"), input_size_tuple):
                    candidates = df[cond & df["尺寸(mm)"].apply(lambda s: size_within_limit(s, input_size_tuple))]
                    if not candidates.empty:
                        result = candidates.iloc[0].to_dict()
                        # 字段补全同上
                        if not result.get("锂电池型号"):
                            for alt in ["推荐电池型号", "型号", "电池型号"]:
                                if result.get(alt):
                                    result["锂电池型号"] = result[alt]
                                    break
                        for field in [
                            "型号", "锂电池型号", "电压(V)", "对应铅酸电池电压(V)", "容量(Ah)", "单体电芯容量(Ah)", "尺寸(mm)", "总重量(kg)", "配重(kg)", "适用叉车型号"
                        ]:
                            if field not in result:
                                result[field] = "-"
            if input_weight > 0:
                bat_weight = float(result.get("总重量(kg)", 0) or 0)
                if bat_weight < input_weight:
                    result["含配重(kg)"] = int(round(input_weight - bat_weight + float(result.get("配重(kg)", 0) or 0)))
                    result["总重量(kg)"] = int(round(input_weight))
                else:
                    result["含配重(kg)"] = int(round(result.get("配重(kg)", 0) or 0))
                    result["总重量(kg)"] = int(round(bat_weight))
            else:
                result["含配重(kg)"] = int(round(result.get("配重(kg)", 0) or 0))
                result["总重量(kg)"] = int(round(result.get("总重量(kg)", 0) or 0))
            code_voltage = int(result["对应铅酸电池电压(V)"])
            final_capacity = int(result["容量(Ah)"])
            counter_weight = float(result.get("配重(kg)", 0) or 0)
            hz_price = 230 * code_voltage * final_capacity / 1000 + counter_weight * 1.5
            nl_price = hz_price * 1.2 / EUR_USD_RATE
            result["惠州出厂价(USD)"] = f"{hz_price:.2f}"
            result["荷兰EXW出货价(EUR)"] = f"{nl_price:.2f}"
            result["汇率(USD/EUR)"] = f"1 EUR = {EUR_USD_RATE:.4f} USD"
            for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)", "汇率(USD/EUR)"]:
                v = result.pop(k)
                result[k] = v
            return result
    elif "原电池类型" in input_data and input_data["原电池类型"] == "铅酸电池":
        target_capacity = input_data.get("容量(Ah)", 0) * 0.75
        cell_capacity = min(CELL_CAPACITIES, key=lambda x: abs(x - target_capacity))
        n_cells = int(np.ceil(target_capacity / cell_capacity))
        final_capacity = n_cells * cell_capacity
        cond = (
            (np.isclose(df["电压(V)"], input_data.get("电压(V)", 0), atol=1))
            & (np.isclose(df["容量(Ah)"], final_capacity, atol=10))
        )
        match = df[cond]
        if not match.empty:
            result = match.iloc[0].to_dict()
            # 补全所有推荐字段
            if not result.get("锂电池型号"):
                for alt in ["推荐电池型号", "型号", "电池型号"]:
                    if result.get(alt):
                        result["锂电池型号"] = result[alt]
                        break
            for field in [
                "型号", "锂电池型号", "电压(V)", "对应铅酸电池电压(V)", "容量(Ah)", "单体电芯容量(Ah)", "尺寸(mm)", "总重量(kg)", "配重(kg)", "适用叉车型号"
            ]:
                if field not in result:
                    result[field] = "-"
            # 尺寸和重量约束过滤
            if input_size_tuple:
                if not size_within_limit(result.get("尺寸(mm)", "-"), input_size_tuple):
                    candidates = df[cond & df["尺寸(mm)"].apply(lambda s: size_within_limit(s, input_size_tuple))]
                    if not candidates.empty:
                        result = candidates.iloc[0].to_dict()
                        # 字段补全同上
                        if not result.get("锂电池型号"):
                            for alt in ["推荐电池型号", "型号", "电池型号"]:
                                if result.get(alt):
                                    result["锂电池型号"] = result[alt]
                                    break
                        for field in [
                            "型号", "锂电池型号", "电压(V)", "对应铅酸电池电压(V)", "容量(Ah)", "单体电芯容量(Ah)", "尺寸(mm)", "总重量(kg)", "配重(kg)", "适用叉车型号"
                        ]:
                            if field not in result:
                                result[field] = "-"
            if input_weight > 0:
                bat_weight = float(result.get("总重量(kg)", 0) or 0)
                if bat_weight < input_weight:
                    result["含配重(kg)"] = int(round(input_weight - bat_weight + float(result.get("配重(kg)", 0) or 0)))
                    result["总重量(kg)"] = int(round(input_weight))
                else:
                    result["含配重(kg)"] = int(round(result.get("配重(kg)", 0) or 0))
                    result["总重量(kg)"] = int(round(bat_weight))
            else:
                result["含配重(kg)"] = int(round(result.get("配重(kg)", 0) or 0))
                result["总重量(kg)"] = int(round(result.get("总重量(kg)", 0) or 0))
            code_voltage = int(result["对应铅酸电池电压(V)"])
            final_capacity = int(result["容量(Ah)"])
            counter_weight = float(result.get("配重(kg)", 0) or 0)
            hz_price = 230 * code_voltage * final_capacity / 1000 + counter_weight * 1.5
            nl_price = hz_price * 1.2 / EUR_USD_RATE
            result["惠州出厂价(USD)"] = f"{hz_price:.2f}"
            result["荷兰EXW出货价(EUR)"] = f"{nl_price:.2f}"
            result["汇率(USD/EUR)"] = f"1 EUR = {EUR_USD_RATE:.4f} USD"
            for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)", "汇率(USD/EUR)"]:
                v = result.pop(k)
                result[k] = v
            return result
    # 3. 兜底：返回最接近的
    idx = ((df["电压(V)"] - input_data.get("电压(V)", 0)).abs() +
           (df["容量(Ah)"] - input_data.get("容量(Ah)", 0)).abs()).idxmin()
    result = df.loc[idx].to_dict()
    code_voltage = int(result["对应铅酸电池电压(V)"])
    final_capacity = int(result["容量(Ah)"])
    counter_weight = float(result.get("配重(kg)", 0) or 0)
    hz_price = 230 * code_voltage * final_capacity / 1000 + counter_weight * 1.5
    nl_price = hz_price * 1.2 / EUR_USD_RATE
    result["惠州出厂价(USD)"] = f"{hz_price:.2f}"
    result["荷兰EXW出货价(EUR)"] = f"{nl_price:.2f}"
    result["汇率(USD/EUR)"] = f"1 EUR = {EUR_USD_RATE:.4f} USD"
    result["总重量(kg)"] = int(round(result.get("总重量(kg)", 0) or 0))
    result["含配重(kg)"] = int(round(result.get("配重(kg)", 0) or 0))
    for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)", "汇率(USD/EUR)"]:
        v = result.pop(k)
        result[k] = v
    return result

# 可继续扩展其它业务函数
