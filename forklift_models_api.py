from flask import Flask, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/api/forklift-models', methods=['GET'])
def get_forklift_models():
    csv_path = os.path.join(os.path.dirname(__file__), 'train_data.csv')
    df = pd.read_csv(csv_path, usecols=["适用叉车型号"])
    models = df["适用叉车型号"].dropna().unique().tolist()
    # 去除空字符串和两端空格，去重
    models = list(set([m.strip() for m in models if m and str(m).strip()]))
    models.sort()
    return jsonify(models)

if __name__ == "__main__":
    app.run(port=8081)
