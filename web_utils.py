# web_utils.py
# Web相关工具函数
import os
import requests

def google_search_forklift_model(model_name):
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not search_engine_id:
        return "未配置Google API密钥，无法联网搜索。"
    url = f"https://www.googleapis.com/customsearch/v1?q={model_name}+forklift+specification&key={api_key}&cx={search_engine_id}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "items" in data and data["items"]:
            snippet = data["items"][0].get("snippet", "")
            return snippet
        else:
            return "未在Google搜索到相关叉车型号信息。"
    except Exception as e:
        return f"Google搜索异常: {e}"
