# 叉车锂电池推荐系统

## 项目结构

- `app.py`：Flask 主入口，API 路由
- `battery_recommend.py`：推荐主逻辑，调用工具函数
- `utils.py`：通用工具函数（尺寸解析、数值处理等）
- `index.html`：前端页面
- `train_model.py`：模型训练脚本
- `train_data.csv`/`valid_data.csv`：训练/验证数据
- `tests/`：单元测试目录

## 快速启动
```bash
pip install -r requirements.txt
python3 app.py
# 浏览器访问 http://localhost:8080
```

## 单元测试
```bash
python3 -m unittest discover tests
```

## 推荐API
- POST `/api/recommend`  
  参数：JSON，详见前端表单字段  
  返回：推荐表格HTML及原始推荐结果

## 主要功能
- 支持多品牌、尺寸、重量、容量等多条件推荐
- 尺寸输入前后端全兼容 x/*/×/X 分隔
- 兜底分支、异常处理健壮

## 部署建议
- 支持 Docker 部署（可按需补充 Dockerfile）
- 推荐使用 Linux/WSL 环境

## 常见问题
- 推荐失败多为参数不符或无匹配型号，可适当放宽输入
- 其它问题请联系维护者
