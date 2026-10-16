# 甄嬛传角色对话系统 (Ollama Baseline)

基于《甄嬛传》角色数据的智能对话系统，使用 LoRA 微调技术训练甄嬛角色模型，支持多种交互方式。

## 📁 项目结构

```
chat_huanhuan/
├── application/          # Web应用界面
│   ├── huanhuan_web.py  # Streamlit对话界面
│   └── chat_history/    # 对话历史存储
├── dataScripts/         # 数据处理脚本
│   ├── huanhuan_data_prepare.py  # 训练数据预处理
│   └── download_data.py          # 数据集下载
├── deployment/          # 模型部署
│   ├── FAST_DEPLOYMENT_GUIDE.md  # 快速部署指南
│   ├── Modelfile.huanhuan        # Ollama模型文件
│   └── huanhuan_fast_lora.gguf   # LoRA权重文件
├── mcp_server/          # MCP服务器
│   ├── __init__.py      # 服务器入口
│   ├── __main__.py      # python -m mcp_server 入口
│   └── server.py        # MCP服务器核心逻辑
├── training/            # 模型训练
│   ├── huanhuan_train.py        # 训练脚本
│   ├── huanhuan_config.yaml     # 训练配置
│   ├── huanhuan_config_fast.yaml # 快速训练配置
│   ├── sync_monitor.py          # 训练资源/进度监控
│   ├── models/                  # LoRA训练产物
│   └── logs/                    # 训练日志
├── data/               # 数据目录
│   ├── raw/            # 原始数据（huanhuan.json）
│   └── processed/      # 处理后的 train/validation/test.jsonl
├── llama.cpp/          # git submodule，用于 LoRA 转 GGUF
├── logs/               # 运行日志
├── huanhuan_web.py     # 与 application/ 下同名文件的副本
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明
```

> `llama.cpp` 是 git submodule，克隆后需要执行 `git submodule update --init --recursive` 拉取。

## 🚀 功能模块

### 📱 Web应用 (application)
- **huanhuan_web.py**: 基于 Streamlit 的甄嬛角色对话Web界面
- 支持实时对话、参数调节、聊天历史管理
- 提供直观的用户界面和流式对话体验

### 📊 数据处理 (dataScripts)
- **huanhuan_data_prepare.py**: 甄嬛传训练数据预处理脚本
- **download_data.py**: 从GitHub下载甄嬛传数据集
- 支持数据清洗、格式转换、分割等功能

### 🚀 模型部署 (deployment)
- **FAST_DEPLOYMENT_GUIDE.md**: 详细的快速部署指南
- **Modelfile.huanhuan**: Ollama模型配置文件
- **huanhuan_fast_lora.gguf**: 训练好的LoRA权重文件
- 支持一键部署到Ollama服务

### 🔌 MCP服务器 (mcp_server)
- **server.py**: MCP (Model Context Protocol) 服务器实现
- 提供与甄嬛模型交互的API接口
- 支持对话、模型信息查询、状态检查等功能

### 🎯 模型训练 (training)
- **huanhuan_train.py**: 甄嬛角色模型训练脚本
- **huanhuan_config.yaml**: 完整训练配置
- **huanhuan_config_fast.yaml**: 快速训练配置
- 基于LoRA技术进行高效微调
- 支持GPU/MPS/CPU多种设备

## 📦 安装依赖

### 方式一：使用 pip 安装

```bash
pip install -r requirements.txt
```

### 方式二：使用 conda 环境（推荐）

```bash
# 1. 创建conda环境（指定Python版本）
conda create -n huanhuan python=3.13

# 2. 激活环境
conda activate huanhuan

# 3. 安装依赖
pip install -r requirements.txt
# 或者优先使用conda安装
conda install pytorch transformers -c pytorch -c huggingface
pip install -r requirements.txt

# 4. 退出环境
conda deactivate
```

### 方式三：使用 uv

```bash
# 1. 安装uv（如果未安装）
pip install uv

# 2. 创建虚拟环境
uv venv huanhuan_env

# 3. 激活环境
source huanhuan_env/bin/activate  # macOS/Linux

# 4. 使用uv安装依赖（比pip快10-100倍）
uv pip install -r requirements.txt
```

## 🏃 运行

以下命令均在仓库根目录执行，需要先启动 Ollama 服务（`ollama serve`，默认 `http://localhost:11434`）。

### 1. 部署甄嬛模型到 Ollama

`deployment/Modelfile.huanhuan` 基于 `qwen2.5:0.5b` 加载 LoRA 权重 `huanhuan_fast_lora.gguf`：

```bash
# 拉取基础模型
ollama pull qwen2.5:0.5b

# 创建甄嬛模型（Modelfile 里的 ADAPTER 是相对路径，需在其所在目录执行）
cd deployment
ollama create huanhuan-qwen -f Modelfile.huanhuan
cd ..

# 命令行验证
ollama run huanhuan-qwen
```

### 2. Web 对话界面

```bash
streamlit run application/huanhuan_web.py
# 指定端口
streamlit run application/huanhuan_web.py --server.port 8501
```

界面默认选用模型 `huanhuan-qwen`，若 Ollama 里没有该名称，可在下拉框中改选已部署的模型。

### 3. MCP 服务器

```bash
python -m mcp_server                      # 默认 stdio
python -m mcp_server sse                  # 或 sse / streamable-http
```

模型名与地址通过环境变量配置：`HUANHUAN_MODEL`（默认 `huanhuan_fast`）、`OLLAMA_HOST`（默认 `http://localhost:11434`）。
注意 MCP 的默认模型名与 Web 界面不同，部署成 `huanhuan-qwen` 后需要显式指定：

```bash
HUANHUAN_MODEL=huanhuan-qwen python -m mcp_server
```

### 4. 训练

```bash
# 下载数据集（来自 datawhalechina/self-llm）
python dataScripts/download_data.py

# 数据预处理
python dataScripts/huanhuan_data_prepare.py

# LoRA 微调（默认读取 ./training/huanhuan_config_fast.yaml，故需在仓库根目录执行）
python training/huanhuan_train.py
```

基础模型为 `Qwen/Qwen2.5-0.5B`，训练产物输出到 `training/models/`。
转成 Ollama 可用的 GGUF 需要 `llama.cpp` 子模块里的 `convert_lora_to_gguf.py`。


