# battery_recommend.py
# 业务逻辑与数据处理
import pandas as pd
import numpy as np
import re
import os
from ai_utils import openai_search_forklift_model

train_df = pd.read_csv("train_data.csv")
df = train_df  # 保持后续兼容
VOLTAGE_MAP = dict(zip(train_df["电压(V)"], train_df["对应铅酸电池电压(V)"]))
CELL_CAPACITIES = sorted(set(train_df["单体电芯容量(Ah)"].dropna().astype(int)))

EUR_USD_RATE = 1.09

def recommend_battery(input_data):
    # 解析原电池尺寸，允许x/X/×分隔，保留原始x分隔用于显示
    input_size = input_data.get("原电池尺寸(mm)", "").replace("×", "x").replace("X", "x")
    # 兼容“数字x数字x数字”无空格格式
    input_size = re.sub(r"(\d)\s*[xX×]\s*(\d)", r"\1x\2", input_size)
    input_size_tuple = None
    if input_size and "x" in input_size:
        try:
            input_size_tuple = tuple(float(x) for x in input_size.split("x"))
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

    # 如果同时输入了叉车品牌和型号，以及电压等参数，自动屏蔽掉品牌和型号，仅用参数分析
    if (
        input_data.get("适用叉车型号") and (
            input_data.get("电压(V)") or input_data.get("容量(Ah)") or input_data.get("总重量(kg)") or input_data.get("原电池尺寸(mm)")
        )
    ):
        input_data = input_data.copy()
        input_data["适用叉车型号"] = ""

    # 1. 叉车型号优先推荐（模糊匹配）
    if "适用叉车型号" in input_data and input_data["适用叉车型号"]:
        match = train_df[train_df["适用叉车型号"] == input_data["适用叉车型号"]]
        if not match.empty:
            candidates = match.copy()
            if input_data.get("电压(V)"):
                candidates = candidates[np.isclose(candidates["电压(V)"], float(input_data["电压(V)"]), atol=2)]
            if input_data.get("容量(Ah)"):
                candidates = candidates[np.isclose(candidates["容量(Ah)"], float(input_data["容量(Ah)"]), atol=10)]
            if input_size_tuple:
                candidates = candidates[candidates["尺寸(mm)"].apply(lambda s: size_within_limit(s, input_size_tuple))]
            if input_data.get("总重量(kg)"):
                candidates = candidates[np.isclose(candidates["总重量(kg)"], float(input_data["总重量(kg)"]), atol=20)]
            if not candidates.empty:
                result = candidates.iloc[0].to_dict()
            else:
                # 有尺寸输入但无合规，直接提示
                if input_size_tuple:
                    return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
                else:
                    return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
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
            # 尺寸和重量约束过滤
            if input_size_tuple:
                if not size_within_limit(result.get("尺寸(mm)", "-"), input_size_tuple):
                    candidates = df[cond & df["尺寸(mm)"].apply(lambda s: size_within_limit(s, input_size_tuple))]
                    if not candidates.empty:
                        result = candidates.iloc[0].to_dict()
                    else:
                        return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
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
            # 推荐结果尺寸格式化为x分隔
            if "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                result["尺寸(mm)"] = result["尺寸(mm)"].replace("×", "x").replace("*", "x").replace("X", "x")

            for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)", "汇率(USD/EUR)"]:
                v = result.pop(k)
                result[k] = v
            # 推荐结果尺寸不能大于输入尺寸（如有输入）
            if input_size_tuple and "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                try:
                    bat_tuple = tuple(float(x) for x in result["尺寸(mm)"].replace("x", "*").split("*"))
                    if len(bat_tuple) == 3 and any(b > l for b, l in zip(sorted(bat_tuple), sorted(input_size_tuple))):
                        return {"推荐失败": "推荐的锂电池尺寸超出原电池尺寸，请调整输入参数或联系技术支持。"}
                except Exception:
                    pass
            return result
        else:
            return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
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
                    else:
                        return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
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
            # 推荐结果尺寸格式化为x分隔
            if "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                result["尺寸(mm)"] = result["尺寸(mm)"].replace("×", "x").replace("*", "x").replace("X", "x")

            for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)", "汇率(USD/EUR)"]:
                v = result.pop(k)
                result[k] = v
            # 推荐结果尺寸不能大于输入尺寸（如有输入）
            if input_size_tuple and "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                try:
                    bat_tuple = tuple(float(x) for x in result["尺寸(mm)"].replace("x", "*").split("*"))
                    if len(bat_tuple) == 3 and any(b > l for b, l in zip(sorted(bat_tuple), sorted(input_size_tuple))):
                        return {"推荐失败": "推荐的锂电池尺寸超出原电池尺寸，请调整输入参数或联系技术支持。"}
                except Exception:
                    pass
            return result
        else:
            return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
    elif "原电池类型" in input_data and input_data["原电池类型"] == "铅酸电池":
        raw_capacity = float(input_data.get("容量(Ah)", 0))
        target_capacity = raw_capacity * 0.75
        if input_size_tuple:
            candidates = df[df["尺寸(mm)"].apply(lambda s: size_within_limit(s, input_size_tuple))]
        else:
            candidates = df
        if not candidates.empty:
            candidates = candidates.copy()
            candidates["容量差"] = (candidates["容量(Ah)"] - target_capacity).abs()
            candidates = candidates[candidates["容量(Ah)"].apply(lambda c: any(abs(c - n*cell) < 1e-2 for cell in CELL_CAPACITIES for n in range(1, 100)))]
            if not candidates.empty:
                best = candidates.sort_values(["容量差"]).iloc[0]
                result = best.to_dict()
                # 字段补全
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
                # 重量补齐
                bat_weight = float(result.get("总重量(kg)", 0) or 0)
                if input_weight > 0:
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
                # 推荐结果尺寸格式化为x分隔
                if "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                    result["尺寸(mm)"] = result["尺寸(mm)"].replace("×", "x").replace("*", "x").replace("X", "x")

                for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)", "汇率(USD/EUR)"]:
                    v = result.pop(k)
                    result[k] = v
                # 推荐结果尺寸不能大于输入尺寸（如有输入）
                if input_size_tuple and "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                    try:
                        bat_tuple = tuple(float(x) for x in result["尺寸(mm)"].replace("x", "*").split("*"))
                        if len(bat_tuple) == 3 and any(b > l for b, l in zip(sorted(bat_tuple), sorted(input_size_tuple))):
                            return {"推荐失败": "推荐的锂电池尺寸超出原电池尺寸，请调整输入参数或联系技术支持。"}
                    except Exception:
                        pass
                return result
            else:
                # 有尺寸输入但所有候选都超限或不符，直接返回友好提示
                return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
        elif input_size_tuple:
            # 有尺寸输入但所有候选都超限，直接返回友好提示
            return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
        else:
            # 兜底：仅无尺寸输入时允许
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
            # 推荐结果尺寸格式化为x分隔
            if "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                result["尺寸(mm)"] = result["尺寸(mm)"].replace("×", "x").replace("*", "x").replace("X", "x")

            for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)", "汇率(USD/EUR)"]:
                v = result.pop(k)
                result[k] = v
            return result

# 可继续扩展其它业务函数
