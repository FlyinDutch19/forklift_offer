# battery_recommend.py
# 业务逻辑与数据处理
import pandas as pd
import numpy as np
import re
import os
from ai_utils import openai_search_forklift_model
from utils import safe_float, parse_battery_size, size_within_limit

train_df = pd.read_csv("train_data.csv")
df = train_df  # 保持后续兼容
VOLTAGE_MAP = dict(zip(train_df["电压(V)"], train_df["对应铅酸电池电压(V)"]))
CELL_CAPACITIES = sorted(set(train_df["单体电芯容量(Ah)"].dropna().astype(int)))

EUR_USD_RATE = 1.09

def recommend_battery(input_data):
    """
    主推荐入口，根据输入参数推荐最优锂电池型号。
    input_data: dict，包含型号、尺寸、容量、品牌等字段
    return: 推荐结果dict，或推荐失败信息
    """
    try:
        # 1. 解析输入尺寸、重量
        # 解析原电池尺寸，允许x/X/×/*分隔，全部转为x
        input_size_tuple = parse_battery_size(input_data.get("原电池尺寸(mm)", ""))
        input_weight = safe_float(input_data.get("总重量(kg)", 0) or 0)

        # 如果同时输入了叉车品牌和型号，以及电压等参数，自动屏蔽掉品牌和型号，仅用参数分析
        if (
            input_data.get("适用叉车型号") and (
                input_data.get("电压(V)") or input_data.get("容量(Ah)") or input_data.get("总重量(kg)") or input_data.get("原电池尺寸(mm)")
            )
        ):
            input_data = input_data.copy()
            input_data["适用叉车型号"] = ""

        # 2. 品牌筛选
        # 电芯品牌筛选，支持“全部”
        cell_brand = input_data.get("电芯品牌")
        if cell_brand and cell_brand != "全部":
            df_brand = df[df["电芯品牌"] == cell_brand].copy()
            if df_brand.empty:
                return {"推荐失败": f"系统中没有{cell_brand}品牌的锂电池型号推荐，建议咨询研发设计人员。"}
        else:
            df_brand = df
        # 后续推荐逻辑全部用df_brand替代df

        # 3. 智能电压映射
        # 智能电压映射：如输入为常见铅酸电池电压（如48、80等），自动映射到最接近的锂电池电压
        if input_data.get("电压(V)"):
            input_voltage = float(input_data["电压(V)"])
            all_voltages = sorted(df["电压(V)"].unique())
            all_lead_voltages = sorted(df["对应铅酸电池电压(V)"].unique())
            # 始终尝试映射：如输入电压与锂电池电压差值大于1，或输入电压正好是常见铅酸电压
            mapped = None
            # 先查映射表
            if int(round(input_voltage)) in all_lead_voltages:
                mapped_mode = df[df["对应铅酸电池电压(V)"] == int(round(input_voltage))]["电压(V)"].mode()
                if not mapped_mode.empty:
                    mapped = float(mapped_mode.iloc[0])
            # 若未命中，直接找最接近的锂电池电压
            if mapped is None:
                mapped = min(all_voltages, key=lambda v: abs(v - input_voltage))
            # 只有当差值大于1才做映射，防止51.2输成51时被强行映射
            if abs(mapped - input_voltage) > 1:
                input_data["电压(V)"] = mapped
                input_voltage = mapped
            # DEBUG: 输出映射后电压
            # print(f"DEBUG: input_data['电压(V)'] after mapping = {input_data.get('电压(V)')}")

        # DEBUG: 打印最终用于筛选的电压
        # print(f"DEBUG: input_data['电压(V)'] = {input_data.get('电压(V)')}")

        # 4. 叉车型号优先推荐（模糊匹配，增强鲁棒性）
        if "适用叉车型号" in input_data and input_data["适用叉车型号"]:
            # 先精确匹配
            model_input = str(input_data["适用叉车型号"]).replace(" ", "").strip().lower()
            df_brand["_型号标准化"] = df_brand["适用叉车型号"].astype(str).replace(" ", "", regex=True).str.strip().str.lower()
            match = df_brand[df_brand["_型号标准化"] == model_input]
            # 若无精确匹配，尝试包含关系的模糊匹配（去空格、统一小写）
            if match.empty:
                match = df_brand[df_brand["_型号标准化"].str.contains(model_input, na=False)]
            if not match.empty:
                candidates = match.copy()
                if input_data.get("电压(V)"):
                    # 电压筛选统一放宽atol=2
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
                        candidates = df_brand[(df_brand["适用叉车型号"] == input_data["适用叉车型号"]) & df_brand["尺寸(mm)"].apply(lambda s: size_within_limit(s, input_size_tuple))]
                        if not candidates.empty:
                            result = candidates.iloc[0].to_dict()
                        else:
                            return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
                if input_weight > 0:
                    bat_weight = safe_float(result.get("总重量(kg)", 0))
                    if bat_weight < input_weight:
                        result["含配重(kg)"] = int(round(input_weight - bat_weight + safe_float(result.get("配重(kg)", 0))))
                        result["总重量(kg)"] = int(round(input_weight))
                    else:
                        result["含配重(kg)"] = int(round(safe_float(result.get("配重(kg)", 0))))
                        result["总重量(kg)"] = int(round(bat_weight))
                else:
                    result["含配重(kg)"] = int(round(safe_float(result.get("配重(kg)", 0))))
                    result["总重量(kg)"] = int(round(safe_float(result.get("总重量(kg)", 0))))
                code_voltage = int(safe_float(result["对应铅酸电池电压(V)"]))
                final_capacity = int(safe_float(result["容量(Ah)"]))
                counter_weight = safe_float(result.get("配重(kg)", 0))
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
                # DEBUG输出所有可选型号，便于排查
                return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
        # 5. 原电池类型与参数推荐
        if "原电池类型" in input_data and input_data["原电池类型"] == "锂电池":
            cond = (
                (np.isclose(df_brand["电压(V)"], input_data.get("电压(V)", 0), atol=2))
                & (np.isclose(df_brand["容量(Ah)"], input_data.get("容量(Ah)", 0), atol=5))
                & (np.isclose(df_brand["总重量(kg)"], input_data.get("总重量(kg)", 0), atol=5))
            )
            match = df_brand[cond]
            if not match.empty:
                candidates = match.copy()
                if not candidates.empty:
                    result = candidates.iloc[0].to_dict()
                else:
                    return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
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
                        candidates = df_brand[cond & df_brand["尺寸(mm)"].apply(lambda s: size_within_limit(s, input_size_tuple))]
                        if not candidates.empty:
                            result = candidates.iloc[0].to_dict()
                        else:
                            return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
                if input_weight > 0:
                    bat_weight = safe_float(result.get("总重量(kg)", 0))
                    if bat_weight < input_weight:
                        result["含配重(kg)"] = int(round(input_weight - bat_weight + safe_float(result.get("配重(kg)", 0))))
                        result["总重量(kg)"] = int(round(input_weight))
                    else:
                        result["含配重(kg)"] = int(round(safe_float(result.get("配重(kg)", 0))))
                        result["总重量(kg)"] = int(round(bat_weight))
                else:
                    result["含配重(kg)"] = int(round(safe_float(result.get("配重(kg)", 0))))
                    result["总重量(kg)"] = int(round(safe_float(result.get("总重量(kg)", 0))))
                code_voltage = int(safe_float(result["对应铅酸电池电压(V)"]))
                final_capacity = int(safe_float(result["容量(Ah)"]))
                counter_weight = safe_float(result.get("配重(kg)", 0))
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
            input_voltage = float(input_data.get("电压(V)", 0))
            if input_size_tuple:
                candidates = df_brand[
                    (np.isclose(df_brand["电压(V)"], input_voltage, atol=2)) &
                    df_brand["尺寸(mm)"].apply(lambda s: size_within_limit(s, input_size_tuple))
                ]
            else:
                candidates = df_brand[np.isclose(df_brand["电压(V)"], input_voltage, atol=2)]
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
                    bat_weight = safe_float(result.get("总重量(kg)", 0))
                    if input_weight > 0:
                        if bat_weight < input_weight:
                            result["含配重(kg)"] = int(round(input_weight - bat_weight + safe_float(result.get("配重(kg)", 0))))
                            result["总重量(kg)"] = int(round(input_weight))
                        else:
                            result["含配重(kg)"] = int(round(safe_float(result.get("配重(kg)", 0))))
                            result["总重量(kg)"] = int(round(bat_weight))
                    else:
                        result["含配重(kg)"] = int(round(safe_float(result.get("配重(kg)", 0))))
                        result["总重量(kg)"] = int(round(safe_float(result.get("总重量(kg)", 0))))
                    code_voltage = int(safe_float(result["对应铅酸电池电压(V)"]))
                    final_capacity = int(safe_float(result["容量(Ah)"]))
                    counter_weight = safe_float(result.get("配重(kg)", 0))
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
                    return {"推荐失败": "系统中没有匹配电压的锂电池型号推荐，建议咨询研发设计人员。"}
            elif input_size_tuple:
                return {"推荐失败": "系统中没有匹配电压的锂电池型号推荐，建议咨询研发设计人员。"}
            else:
                # 仅返回简洁提示，不含DEBUG
                return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
    except Exception as e:
        # 返回友好错误提示，并带 DEBUG 信息
        return {"推荐失败": f"服务异常，请稍后重试。DEBUG: {str(e)}"}

# 可继续扩展其它业务函数
