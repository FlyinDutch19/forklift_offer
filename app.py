from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from battery_recommend import recommend_battery
import logging
import sys

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

@app.route("/")
def index():
    try:
        return app.send_static_file("index.html")
    except Exception:
        return "<h2>index.html 未找到或无权限</h2>", 404

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    input_data = request.json
    try:
        # 处理单体电芯容量可选项
        cell_caps = input_data.get("单体电芯容量可选项")
        if cell_caps:
            try:
                cell_caps_list = [int(x.strip()) for x in cell_caps.split(",") if x.strip().isdigit()]
                input_data["单体电芯容量可选项"] = cell_caps_list
            except Exception:
                input_data["单体电芯容量可选项"] = []
        result = recommend_battery(input_data)
        discount = None
        if "折扣率(%)" in input_data:
            try:
                discount = float(input_data["折扣率(%)"])
            except Exception:
                discount = None
        def format_result_table(result_dict, discount=None):
            show = dict(result_dict)
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
                ("模组配置(串S并P联）", "模组配置(串S并P联）"), # 兼容老字段
                ("尺寸(mm)", "尺寸(mm)"),
                ("总重量(kg)", "总重量(kg)"),
                ("含配重(kg)", "含配重(kg)"),
                ("惠州出厂价(USD)", "惠州出厂价(USD)折后价（不含VAT税）"),
                ("荷兰EXW出货价(EUR)", "荷兰EXW出货价(EUR)折后价（不含VAT税）"),
                ("汇率(USD/EUR)", "汇率(USD/EUR)")
            ]
            table = '<table style="border-collapse:separate;border-spacing:0 8px;min-width:420px;width:80%;">'
            for k, k2 in field_map:
                # 模组串并联方式优先显示新字段，否则用老字段
                if k == "模组串并联方式":
                    v = show.get("模组串并联方式") or show.get("模组配置(串S并P联）") or "-"
                # 价格字段显示折扣价
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
                # 跳过不存在的老字段
                if k == "模组配置(串S并P联）":
                    continue
                table += f'<tr><th style="text-align:right;vertical-align:top;font-weight:bold;padding:8px 18px 8px 0;background:#f6f6f6;font-size:16px;width:220px;">{k2}</th>' \
                         f'<td style="text-align:left;vertical-align:top;font-weight:normal;padding:8px 0 8px 8px;font-size:16px;">{v}</td></tr>'
            table += '</table>'
            return table
        # 推荐失败友好提示分支
        if result is None:
            return jsonify({"error": "系统异常，未能获取推荐结果。"}), 500
        if isinstance(result, dict) and "推荐失败" in result:
            logging.info(f"[RECOMMEND FAIL] input={input_data} response={result}")
            from flask import make_response
            import json
            resp = make_response(json.dumps(result, ensure_ascii=False, allow_nan=False), 200)
            resp.headers['Content-Type'] = 'application/json; charset=utf-8'
            return resp
        # 兼容推荐结果为多条（dict嵌套dict）
        if isinstance(result, dict) and all(isinstance(v, dict) for v in result.values()):
            tables = []
            raw_results = {}
            for idx, (k, v) in enumerate(result.items(), 1):
                # 处理所有NaN为None，避免json.dumps出错
                import numpy as np
                for kk, vv in v.items():
                    if isinstance(vv, float) and (np.isnan(vv) or vv is None):
                        v[kk] = None
                    if vv == 'nan':
                        v[kk] = None
                table_html = format_result_table(v, discount)
                tables.append(f'<h4 style="margin-top:18px;">{k}</h4>' + table_html)
                raw_results[k] = v
            from flask import make_response
            import json
            resp = make_response(json.dumps({"table": "<br>".join(tables), "raw": raw_results}, ensure_ascii=False, allow_nan=False), 200)
            resp.headers['Content-Type'] = 'application/json; charset=utf-8'
            return resp
        # 兼容AI推荐分支
        if isinstance(result, dict):
            if "推荐电池型号" in result:
                result["锂电池型号"] = result.pop("推荐电池型号")
            # 处理所有NaN为None，避免json.dumps出错
            import numpy as np
            for k, v in result.items():
                if isinstance(v, float) and (np.isnan(v) or v is None):
                    result[k] = None
                if v == 'nan':
                    result[k] = None
            table_html = format_result_table(result, discount)
            logging.info(f"[RECOMMEND OK] input={input_data} response=table+raw")
            from flask import make_response
            import json
            resp = make_response(json.dumps({"table": table_html, "raw": result}, ensure_ascii=False, allow_nan=False), 200)
            resp.headers['Content-Type'] = 'application/json; charset=utf-8'
            return resp
        # 兼容极端情况（如返回列表等）
        if isinstance(result, list):
            table_html = format_result_table({k: v for k, v in enumerate(result)}, discount)
            logging.info(f"[RECOMMEND OK] input={input_data} response=table+raw(list)")
            from flask import make_response
            import json
            resp = make_response(json.dumps({"table": table_html, "raw": result}, ensure_ascii=False, allow_nan=False), 200)
            resp.headers['Content-Type'] = 'application/json; charset=utf-8'
            return resp
        # 其它未知类型，直接返回错误
        return jsonify({"error": "系统异常，未能获取推荐结果。"}), 500
    except Exception as e:
        import traceback
        logging.error(f"[RECOMMEND ERROR] input={input_data} error={e} trace={traceback.format_exc()}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/forklift-models", methods=["GET"])
def api_forklift_models():
    # 优先从all_forklift_models.txt加载全集合
    import os
    txt_path = os.path.join(os.path.dirname(__file__), 'all_forklift_models.txt')
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            models = [line.strip() for line in f if line.strip() and line.strip() != "N/A"]
        models = sorted(set(models))
        return jsonify(models)
    # 兜底：从train_data.csv加载
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), 'train_data.csv')
    df = pd.read_csv(csv_path, usecols=["适用叉车型号"])
    models = df["适用叉车型号"].dropna().unique().tolist()
    models = list(set([m.strip() for m in models if m and str(m).strip() and m != "N/A"]))
    models.sort()
    return jsonify(models)

if __name__ == "__main__":
    # 生产环境下用debug=False，避免reloader导致nohup后台运行报错
    app.run(debug=False, host="0.0.0.0", port=8080)
