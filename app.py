from flask import Flask, request, jsonify
from flask_cors import CORS
from battery_recommend import recommend_battery
import html
import logging
import sys
import math
import numpy as np
import pandas as pd
import requests

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def safe_str(val):
    if val is None:
        return "-"
    if isinstance(val, float):
        if val != val:  # NaN
            return "-"
        return f"{val:.2f}"
    if isinstance(val, (int, str)):
        return str(val)
    return str(val)

def clean_json(obj):
    # 递归清理所有 NaN/None/np.nan/pd.NA/字符串'nan'
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif obj is None:
        return "-"
    elif isinstance(obj, float):
        if math.isnan(obj):
            return "-"
        return obj
    elif str(obj) == "nan":
        return "-"
    try:
        import numpy as np
        import pandas as pd
        if obj is np.nan or obj is pd.NA:
            return "-"
    except Exception:
        pass
    return obj

def format_result_table(result_dict, discount=None):
    # 先递归清理所有 NaN/None
    show = clean_json(result_dict)
    # 字段顺序与映射
    field_map = [
        ("适用叉车型号", "适用叉车型号"),
        ("锂电池型号", "锂电池型号"),
        ("电芯品牌", "电芯品牌"),
        ("电压(V)", "电压(V)"),
        ("对应铅酸电池电压(V)", "对应铅酸电池电压(V)"),
        ("容量(Ah)", "容量(Ah)"),
        ("单体电芯容量(Ah)", "单体电芯容量(Ah)"),
        ("模组串并联方式", "模组串并联方式"),
        ("模组配置(串S并P联）", "模组配置(串S并P联）"),
        ("尺寸(mm)", "尺寸(mm)"),
        ("总重量(kg)", "总重量(kg)"),
        ("含配重(kg)", "含配重(kg)"),
        ("惠州出厂价(USD)", "惠州出厂价(USD)折后价（不含VAT税）"),
        ("荷兰EXW出货价(EUR)", "荷兰EXW出货价(EUR)折后价（不含VAT税）"),
        ("汇率(EUR/USD)", "汇率(EUR/USD)")
    ]
    table = '<table style="border-collapse:separate;border-spacing:0 8px;min-width:420px;width:80%;">'
    for k, k2 in field_map:
        if k == "模组串并联方式":
            v1 = show.get("模组串并联方式")
            v2 = show.get("模组配置(串S并P联）")
            v = next((x for x in [v1, v2] if x not in [None, "", "nan", "-", "None"]), "-")
        elif k == "惠州出厂价(USD)":
            try:
                hz = float(show.get("惠州出厂价(USD)", 0))
                hz_discount = hz * discount / 100 if discount is not None else hz
                v = f"{hz_discount:.2f}"
            except Exception:
                v = "-"
        elif k == "荷兰EXW出货价(EUR)":
            try:
                nl = float(show.get("荷兰EXW出货价(EUR)", 0))
                nl_discount = nl * discount / 100 if discount is not None else nl
                v = f"{nl_discount:.2f}"
            except Exception:
                v = "-"
        else:
            v = show.get(k) if show.get(k) not in [None, "", "nan"] else "-"
        if k == "模组配置(串S并P联）":
            continue
        k2 = html.escape(safe_str(k2).replace('{', '').replace('}', ''))
        v = html.escape(safe_str(v).replace('{', '').replace('}', ''))
        table += '<tr><th style="text-align:right;vertical-align:top;font-weight:bold;padding:8px 18px 8px 0;background:#f6f6f6;font-size:16px;width:220px;">' + k2 + '</th><td style="text-align:left;vertical-align:top;font-weight:normal;padding:8px 0 8px 8px;font-size:16px;">' + v + '</td></tr>'
    table += '</table>'
    return table

def get_eur_usd_rate():
    """实时获取欧元对美元汇率（EUR/USD），失败时返回默认值并记录日志"""
    try:
        resp = requests.get("https://api.exchangerate.host/latest?base=EUR&symbols=USD", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            rate = data.get("rates", {}).get("USD")
            if rate:
                return float(rate)
    except Exception as e:
        logging.warning(f"[汇率获取失败] {e}")
    return 1.08  # 默认值，可根据实际情况调整

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    try:
        input_data = request.json
        try:
            cell_caps = input_data.get("单体电芯容量可选项")
            if cell_caps:
                try:
                    cell_caps_list = [int(x.strip()) for x in cell_caps.split(",") if x.strip().isdigit()]
                    input_data["单体电芯容量可选项"] = cell_caps_list
                except Exception:
                    input_data["单体电芯容量可选项"] = []
            discount = None
            if "折扣率(%)" in input_data:
                try:
                    discount = float(input_data["折扣率(%)"])
                except Exception:
                    discount = None
            # 实时获取汇率
            eur_usd_rate = get_eur_usd_rate()
            input_data["汇率(EUR/USD)"] = eur_usd_rate
            # 记录输入
            with open('flask.log', 'a', encoding='utf-8') as f:
                f.write("\n[RECOMMEND INPUT] " + str(input_data) + "\n")
            result = recommend_battery(input_data)
            # 记录输出
            with open('flask.log', 'a', encoding='utf-8') as f:
                f.write("[RECOMMEND OUTPUT] " + str(result) + "\n")
            # 推荐失败
            if result is None or (isinstance(result, dict) and "推荐失败" in result):
                msg = result["推荐失败"] if isinstance(result, dict) and "推荐失败" in result else "系统中没有匹配的锂电池型号推荐，建议咨询研发设计人员。"
                return jsonify({"error": msg}), 200
            # 多条推荐
            if isinstance(result, dict) and all(isinstance(v, dict) for v in result.values()):
                tables = []
                for k, v in result.items():
                    v["汇率(EUR/USD)"] = f"{eur_usd_rate:.4f}"
                    tables.append('<h4 style="margin-top:18px;">' + html.escape(safe_str(k)) + '</h4>' + format_result_table(v, discount))
                return jsonify({"table": "<br>".join(tables), "raw": clean_json(result)})
            # 单条推荐
            if isinstance(result, dict):
                if "推荐电池型号" in result:
                    result["锂电池型号"] = result.pop("推荐电池型号")
                result["汇率(EUR/USD)"] = f"{eur_usd_rate:.4f}"
                table_html = format_result_table(result, discount)
                return jsonify({"table": table_html, "raw": clean_json(result)})
            # 列表推荐
            if isinstance(result, list):
                for v in result:
                    if isinstance(v, dict):
                        v["汇率(EUR/USD)"] = f"{eur_usd_rate:.4f}"
                table_html = format_result_table({k: v for k, v in enumerate(result)}, discount)
                return jsonify({"table": table_html, "raw": clean_json(result)})
            return jsonify({"error": "系统异常，未能获取推荐结果。"}), 500
        except Exception as e:
            import traceback
            logging.error("[RECOMMEND ERROR] input=%s error=%s trace=%s", input_data, e, traceback.format_exc())
            with open('flask.log', 'a', encoding='utf-8') as f:
                f.write("\n[RECOMMEND ERROR] input= " + str(input_data) + "\n")
                f.write("[RECOMMEND ERROR] error= " + str(e) + "\n")
                f.write("[RECOMMEND ERROR] trace=\n" + traceback.format_exc() + "\n")
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    except Exception as e:
        import traceback
        with open('flask.log', 'a', encoding='utf-8') as f:
            f.write("\n[RECOMMEND FATAL] error= " + str(e) + "\n")
            f.write("[RECOMMEND FATAL] trace=\n" + traceback.format_exc() + "\n")
        return jsonify({"error": "fatal: " + str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/forklift-models", methods=["GET"])
def api_forklift_models():
    import os
    txt_path = os.path.join(os.path.dirname(__file__), 'all_forklift_models.txt')
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            models = [line.strip() for line in f if line.strip() and line.strip() != "N/A"]
        models = sorted(set(models))
        return jsonify(models)
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), 'train_data.csv')
    df = pd.read_csv(csv_path, usecols=["适用叉车型号"])
    models = df["适用叉车型号"].dropna().unique().tolist()
    models = list(set([m.strip() for m in models if m and str(m).strip() and m != "N/A"]))
    models.sort()
    return jsonify(models)

@app.route("/")
def index():
    try:
        return app.send_static_file("index.html")
    except Exception:
        return "<h2>index.html 未找到或无权限</h2>", 404

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
