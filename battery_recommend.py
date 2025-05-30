# battery_recommend.py
# 业务逻辑与数据处理
import pandas as pd
import numpy as np
import re
import os
from ai_utils import openai_search_forklift_model
from utils import safe_float, parse_battery_size, size_within_limit

all_df = pd.read_csv("all_data.csv")
df = all_df  # 推荐主数据源
VOLTAGE_MAP = dict(zip(all_df["电压(V)"], all_df["对应铅酸电池电压(V)"]))
CELL_CAPACITIES = sorted(set(all_df["单体电芯容量(Ah)"].dropna().astype(int)))

EUR_USD_RATE = 1.09

def recommend_battery(input_data, _is_fallback=False):
    """
    主推荐入口，根据输入参数推荐最优锂电池型号。
    input_data: dict，包含型号、尺寸、容量、品牌等字段
    return: 推荐结果dict，或推荐失败信息
    """
    try:
        # 0. 读取汇率，优先用 input_data 传入的 EUR/USD 汇率
        eur_usd_rate = None
        for k in ["汇率(EUR/USD)", "EUR/USD", "eur_usd_rate"]:
            if k in input_data:
                try:
                    eur_usd_rate = float(input_data[k])
                    break
                except Exception:
                    pass
        if not eur_usd_rate or eur_usd_rate <= 0:
            eur_usd_rate = EUR_USD_RATE

        # 1. 解析输入尺寸、重量
        # 解析原电池尺寸，允许x/X/×/*分隔，全部转为x
        input_size_tuple = parse_battery_size(input_data.get("原电池尺寸(mm)", ""))
        input_weight = safe_float(input_data.get("总重量(kg)", 0) or 0)

        def is_effective(val):
            try:
                return float(val) > 0
            except Exception:
                return bool(val and str(val).strip())

        # 如果同时输入了叉车品牌和型号，以及电压等参数，自动屏蔽掉品牌和型号，仅用参数分析
        if (
            input_data.get("适用叉车型号") and (
                is_effective(input_data.get("电压(V)")) or
                is_effective(input_data.get("容量(Ah)")) or
                is_effective(input_data.get("总重量(kg)")) or
                is_effective(input_data.get("原电池尺寸(mm)"))
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

        # 4. 叉车型号模糊推荐（极宽松，包含即出）
        if "适用叉车型号" in input_data and input_data["适用叉车型号"]:
            model_input = str(input_data["适用叉车型号"]).replace(" ", "").strip().lower()
            df_brand["_型号标准化"] = df_brand["适用叉车型号"].astype(str).replace(" ", "", regex=True).str.strip().str.lower()
            # 只要包含输入字符串的都输出
            match = df_brand[df_brand["_型号标准化"].str.contains(model_input, na=False)]
            if not match.empty:
                candidates = match.copy()
                results = {}
                for idx, row in candidates.iterrows():
                    result = row.to_dict()
                    # 字段补全
                    if not result.get("锂电池型号"):
                        for alt in ["推荐电池型号", "型号", "电池型号"]:
                            if result.get(alt):
                                result["锂电池型号"] = result[alt]
                                break
                    for field in [
                        "锂电池型号", "电芯品牌", "电压(V)", "对应铅酸电池电压(V)", "容量(Ah)", "单体电芯容量(Ah)", "尺寸(mm)", "总重量(kg)", "配重(kg)", "适用叉车型号"
                    ]:
                        if field not in result or pd.isna(result[field]) or result[field] == '':
                            if field in ["容量(Ah)", "单体电芯容量(Ah)", "电压(V)", "对应铅酸电池电压(V)", "总重量(kg)", "配重(kg)"]:
                                result[field] = 0
                            else:
                                result[field] = "-"
                    # 电芯品牌补全
                    if not result.get("电芯品牌") and result.get("锂电池型号"):
                        brand_row = df[df["锂电池型号"] == result["锂电池型号"]]
                        if not brand_row.empty:
                            result["电芯品牌"] = brand_row.iloc[0]["电芯品牌"]
                    # 含配重优先用原始数据
                    if "含配重(kg)" in row and safe_float(row["含配重(kg)"]) > 0:
                        result["含配重(kg)"] = int(round(safe_float(row["含配重(kg)"])))
                    else:
                        result["含配重(kg)"] = int(round(safe_float(result.get("配重(kg)", 0))))
                    # 价格等字段补全
                    code_voltage = int(safe_float(result["对应铅酸电池电压(V)"]))
                    final_capacity = int(safe_float(result["容量(Ah)"]))
                    counter_weight = safe_float(result.get("配重(kg)", 0))
                    # 价格基数调整：如输入有惠州出厂价和配重价则用输入值，否则用默认
                    hz_base = safe_float(input_data.get("惠州出厂价(USD)（不含VAT税）", 230))  # 单位USD/KWH
                    hz_weight_base = safe_float(input_data.get("惠州配重出厂价(USD)（不含VAT税）", 1.5))  # 单位USD/KG
                    # 计算KWH
                    kwh = safe_float(result.get("电压(V)", 0)) * safe_float(result.get("容量(Ah)", 0)) / 1000
                    hz_price = hz_base * kwh + counter_weight * hz_weight_base
                    nl_price = hz_price * 1.2 / eur_usd_rate
                    result["惠州出厂价(USD)"] = f"{hz_price:.2f}"
                    result["荷兰EXW出货价(EUR)"] = f"{nl_price:.2f}"
                    # 删除 result["汇率(USD/EUR)"] 字段
                    # 尺寸格式化
                    if "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                        size_str = result["尺寸(mm)"].replace("×", "x").replace("*", "x").replace("X", "x")
                        parts = [p.strip() for p in size_str.split("x") if p.strip()]
                        while len(parts) < 3:
                            parts.append("-")
                        result["尺寸(mm)"] = "x".join(parts[:3])
                    for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)"]:
                        v = result.pop(k)
                        result[k] = v
                    results[f"推荐结果{len(results)+1}"] = result
                if results:
                    return results
                else:
                    return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
        # 5. 原电池类型与参数推荐
        if "原电池类型" in input_data and input_data["原电池类型"] == "锂电池":
            cond = np.ones(len(df_brand), dtype=bool)
            input_voltage = None
            if input_data.get("电压(V)"):
                input_voltage = float(input_data["电压(V)"])
                # 先尝试精确匹配
                cond &= np.isclose(df_brand["电压(V)"], input_voltage, atol=2)
            if input_data.get("容量(Ah)"):
                cond &= np.isclose(df_brand["容量(Ah)"], input_data.get("容量(Ah)", 0), atol=5)
            if input_data.get("总重量(kg)"):
                cond &= np.isclose(df_brand["总重量(kg)"], input_data.get("总重量(kg)", 0), atol=5)
            match = df_brand[cond]
            # 智能电压映射：如无精确匹配且输入电压为常见铅酸电压，则自动映射到最接近的锂电池电压
            if (input_voltage is not None) and match.empty:
                all_voltages = sorted(df_brand["电压(V)"].unique())
                mapped_voltage = min(all_voltages, key=lambda v: abs(v - input_voltage))
                # 只有当差值大于1才做映射，防止51.2输成51时被强行映射
                if abs(mapped_voltage - input_voltage) > 1:
                    cond2 = np.ones(len(df_brand), dtype=bool)
                    cond2 &= np.isclose(df_brand["电压(V)"], mapped_voltage, atol=2)
                    if input_data.get("容量(Ah)"):
                        cond2 &= np.isclose(df_brand["容量(Ah)"], input_data.get("容量(Ah)", 0), atol=5)
                    if input_data.get("总重量(kg)"):
                        cond2 &= np.isclose(df_brand["总重量(kg)"], input_data.get("总重量(kg)", 0), atol=5)
                    match = df_brand[cond2]
            if not match.empty:
                candidates = match.copy()
                results = {}
                for idx, row in candidates.head(3).iterrows():
                    # 先标准化字段名，去除所有key的前后空格
                    result = {k.strip(): v for k, v in row.to_dict().items()}
                    # 字段补全
                    if not result.get("锂电池型号"):
                        for alt in ["推荐电池型号", "型号", "电池型号"]:
                            if result.get(alt):
                                result["锂电池型号"] = result[alt]
                                break
                    # 字段补全，仅补全实际业务需要的字段，且只在缺失或为NaN时赋默认值
                    for field in [
                        "锂电池型号", "电芯品牌", "电压(V)", "对应铅酸电池电压(V)", "容量(Ah)", "单体电芯容量(Ah)", "尺寸(mm)", "总重量(kg)", "配重(kg)", "适用叉车型号"
                    ]:
                        if field not in result or pd.isna(result[field]) or result[field] == '':
                            if field in ["容量(Ah)", "单体电芯容量(Ah)", "电压(V)", "对应铅酸电池电压(V)", "总重量(kg)", "配重(kg)"]:
                                result[field] = 0
                            else:
                                result[field] = "-"
                    # 电芯品牌补全
                    if not result.get("电芯品牌") and result.get("锂电池型号"):
                        # 从原数据查找电芯品牌
                        brand_row = df[df["锂电池型号"] == result["锂电池型号"]]
                        if not brand_row.empty:
                            result["电芯品牌"] = brand_row.iloc[0]["电芯品牌"]
                    if input_size_tuple:
                        if not size_within_limit(result.get("尺寸(mm)", "-"), input_size_tuple):
                            continue
                    # 含配重优先用原始数据
                    if "含配重(kg)" in row and safe_float(row["含配重(kg)"]) > 0:
                        result["含配重(kg)"] = int(round(safe_float(row["含配重(kg)"])))
                    elif input_weight > 0:
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
                    # 价格基数调整：如输入有惠州出厂价和配重价则用输入值，否则用默认
                    hz_base = safe_float(input_data.get("惠州出厂价(USD)（不含VAT税）", 230))  # 单位USD/KWH
                    hz_weight_base = safe_float(input_data.get("惠州配重出厂价(USD)（不含VAT税）", 1.5))  # 单位USD/KG
                    # 计算KWH
                    kwh = safe_float(result.get("电压(V)", 0)) * safe_float(result.get("容量(Ah)", 0)) / 1000
                    counter_weight = safe_float(result.get("配重(kg)", 0))
                    hz_price = hz_base * kwh + counter_weight * hz_weight_base
                    nl_price = hz_price * 1.2 / eur_usd_rate
                    result["惠州出厂价(USD)"] = f"{hz_price:.2f}"
                    result["荷兰EXW出货价(EUR)"] = f"{nl_price:.2f}"
                    # 删除 result["汇率(USD/EUR)"] 字段
                    if "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                        size_str = result["尺寸(mm)"].replace("×", "x").replace("*", "x").replace("X", "x")
                        parts = [p.strip() for p in size_str.split("x") if p.strip()]
                        while len(parts) < 3:
                            parts.append("-")
                        result["尺寸(mm)"] = "x".join(parts[:3])
                    for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)"]:
                        v = result.pop(k)
                        result[k] = v
                    # 推荐结果尺寸不能大于输入尺寸（如有输入）
                    if input_size_tuple and "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                        try:
                            bat_tuple = tuple(float(x) for x in result["尺寸(mm)"].replace("x", "*").split("*"))
                            if len(bat_tuple) == 3 and any(b > l for b, l in zip(sorted(bat_tuple), sorted(input_size_tuple))):
                                continue
                        except Exception:
                            pass
                    results[f"推荐结果{len(results)+1}"] = result
                if results:
                    return results
                else:
                    return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
            else:
                return {"推荐失败": "系统中没有匹配电压的锂电池型号推荐，建议咨询研发设计人员。"}
        elif "原电池类型" in input_data and input_data["原电池类型"] == "铅酸电池":
            raw_capacity = float(input_data.get("容量(Ah)", 0))
            target_capacity = raw_capacity * 0.8  # 修改为0.8
            input_voltage = float(input_data.get("电压(V)", 0))
            # 智能电压映射：如输入电压与锂电池电压差值大于1，或数据源无精确匹配，则自动映射到最接近的锂电池电压
            all_voltages = sorted(df_brand["电压(V)"].unique())
            # 只有当数据源无精确匹配时才做映射，防止51.2输成51时被强行映射
            if not np.any(np.isclose(all_voltages, input_voltage, atol=1e-2)):
                mapped_voltage = min(all_voltages, key=lambda v: abs(v - input_voltage))
                # 只有当差值大于1才做映射
                if abs(mapped_voltage - input_voltage) > 1:
                    input_voltage = mapped_voltage
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
                    results = {}
                    for idx, row in candidates.sort_values(["容量差"]).head(3).iterrows():
                        result = row.to_dict()
                        # 字段补全
                        if not result.get("锂电池型号"):
                            for alt in ["推荐电池型号", "型号", "电池型号"]:
                                if result.get(alt):
                                    result["锂电池型号"] = result[alt]
                                    break
                        # 字段补全，仅补全实际业务需要的字段，且只在缺失或为NaN时赋默认值
                        for field in [
                            "锂电池型号", "电芯品牌", "电压(V)", "对应铅酸电池电压(V)", "容量(Ah)", "单体电芯容量(Ah)", "尺寸(mm)", "总重量(kg)", "配重(kg)", "适用叉车型号"
                        ]:
                            if field not in result or pd.isna(result[field]) or result[field] == '':
                                if field in ["容量(Ah)", "单体电芯容量(Ah)", "电压(V)", "对应铅酸电池电压(V)", "总重量(kg)", "配重(kg)"]:
                                    result[field] = 0
                                else:
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
                        # 价格基数调整：如输入有惠州出厂价和配重价则用输入值，否则用默认
                        hz_base = safe_float(input_data.get("惠州出厂价(USD)（不含VAT税）", 230))  # 单位USD/KWH
                        hz_weight_base = safe_float(input_data.get("惠州配重出厂价(USD)（不含VAT税）", 1.5))  # 单位USD/KG
                        # 计算KWH
                        kwh = safe_float(result.get("电压(V)", 0)) * safe_float(result.get("容量(Ah)", 0)) / 1000
                        counter_weight = safe_float(result.get("配重(kg)", 0))
                        hz_price = hz_base * kwh + counter_weight * hz_weight_base
                        nl_price = hz_price * 1.2 / eur_usd_rate
                        result["惠州出厂价(USD)"] = f"{hz_price:.2f}"
                        result["荷兰EXW出货价(EUR)"] = f"{nl_price:.2f}"
                        # 删除 result["汇率(USD/EUR)"] 字段
                        if "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                            size_str = result["尺寸(mm)"].replace("×", "x").replace("*", "x").replace("X", "x")
                            parts = [p.strip() for p in size_str.split("x") if p.strip()]
                            while len(parts) < 3:
                                parts.append("-")
                            result["尺寸(mm)"] = "x".join(parts[:3])
                        for k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)"]:
                            v = result.pop(k)
                            result[k] = v
                        if input_size_tuple and "尺寸(mm)" in result and isinstance(result["尺寸(mm)"], str):
                            try:
                                bat_tuple = tuple(float(x) for x in result["尺寸(mm)"].replace("x", "*").split("*"))
                                if len(bat_tuple) == 3 and any(b > l for b, l in zip(sorted(bat_tuple), sorted(input_size_tuple))):
                                    continue
                            except Exception:
                                pass
                        results[f"推荐结果{len(results)+1}"] = result
                    if results:
                        return results
                    else:
                        return {"推荐失败": "系统中没有匹配电压的锂电池型号推荐，建议咨询研发设计人员。"}
            elif input_size_tuple:
                return {"推荐失败": "系统中没有匹配电压的锂电池型号推荐，建议咨询研发设计人员。"}
            else:
                return {"推荐失败": "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"}
    except Exception as e:
        # 返回友好错误提示（去除DEBUG信息）
        return {"推荐失败": "服务异常，请稍后重试。"}

# 可继续扩展其它业务函数
