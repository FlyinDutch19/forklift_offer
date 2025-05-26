from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from battery_recommend import recommend_battery

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

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
        result = recommend_battery(input_data)
        def format_result_table(result_dict):
            fields = [
                ("适用叉车型号", "Forklift Model"),
                ("锂电池型号", "Battery Model"),
                ("电压(V)", "Voltage (V)"),
                ("对应铅酸电池电压(V)", "Lead-acid Battery Voltage (V)"),
                ("容量(Ah)", "Capacity (Ah)"),
                ("单体电芯容量(Ah)", "Cell Capacity (Ah)"),
                ("尺寸(mm)", "Dimensions (mm)"),
                ("总重量(kg)", "Total Weight (kg)"),
                ("含配重(kg)", "Counterweight (kg)"),
                ("惠州出厂价(USD)", "Huizhou Price (USD)"),
                ("荷兰EXW出货价(EUR)", "Netherlands EXW (EUR)"),
                ("汇率(USD/EUR)", "USD/EUR Rate")
            ]
            # 字段修正和格式化
            show = dict(result_dict)
            # 锂电池型号兼容推荐电池型号和型号字段
            if (not show.get("锂电池型号")):
                for alt in ["推荐电池型号", "型号", "电池型号"]:
                    if show.get(alt):
                        show["锂电池型号"] = show[alt]
                        break
            # 对应铅酸电池电压(V)字段名修正（兼容所有繁体和简体写法）
            for wrong in ["对应铅酸電池電壓(V)", "对应铅酸電壓(V)", "对应铅酸电池電壓(V)"]:
                if wrong in show:
                    show["对应铅酸电池电压(V)"] = show.pop(wrong)
            # 容量(Ah)不带小数
            if "容量(Ah)" in show:
                try:
                    show["容量(Ah)"] = str(int(float(show["容量(Ah)"])) )
                except Exception:
                    pass
            table = '<table style="border-collapse:separate;border-spacing:0 8px;min-width:520px;width:98%;">'
            for k, _ in fields:
                th = k
                if k in ["惠州出厂价(USD)", "荷兰EXW出货价(EUR)"]:
                    th = f"{k}<span style='color:#888;font-size:12px;'>&nbsp;(不含VAT税费)</span>"
                v = show.get(k, "-")
                table += f'<tr><th style="text-align:right;vertical-align:top;font-weight:bold;padding:8px 18px 8px 0;background:#f6f6f6;font-size:16px;width:220px;">{th}</th>' \
                         f'<td style="text-align:left;vertical-align:top;font-weight:normal;padding:8px 0 8px 8px;font-size:16px;">{v}</td></tr>'
            table += '</table>'
            return table
        # 兼容AI推荐分支
        if isinstance(result, dict):
            if "推荐电池型号" in result:
                result["锂电池型号"] = result.pop("推荐电池型号")
            table_html = format_result_table(result)
            return jsonify({"table": table_html, "raw": result})
        # 兼容极端情况（如返回列表等）
        table_html = format_result_table({k: v for k, v in enumerate(result)})
        return jsonify({"table": table_html, "raw": result})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
