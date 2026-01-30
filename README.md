# PACK_AI

电池包智能分析系统 - 基于机器学习和统计分析的电池包制造质量检测平台

## 项目简介

PACK_AI 是一个综合性的电池包分析系统，旨在通过机器学习和统计分析技术，对电池包的制造和测试数据进行深入分析，帮助识别潜在问题并优化生产流程。

系统采用微服务架构，提供多个专门的分析服务，支持温度相关性分析、DCR（直流内阻）异常检测、测试结果预测和工艺过程分析等功能。

## 核心功能

### 1. 温度相关性分析服务 (`/temp`)

分析电池包温度与能量/容量指标之间的相关性。

- **端点**: `POST /temp/pack-temp-corr`
- **功能**: 计算最低温度与充放电能量/容量之间的皮尔逊相关系数
- **输入**: 车辆代码 (vehicle_code) 和 工步ID (step_id)
- **输出**: 相关系数和时序数据

### 2. DCR分析服务 (`/dcr`)

分析直流内阻（DCR）数据并检测异常电芯。

- **端点**: `POST /dcr/pack-dcr-analysis`
- **功能**:
  - DCR异常检测（基于Z-score）
  - 与制造参数（容量、重量、OCV等）的相关性分析
- **输入**: 电池包编码列表 (pack_code_list)
- **输出**: 异常电芯、DCR值及参数相关性分析结果

### 3. 结果分析服务 (`/result`)

基于机器学习模型进行测试结果分析和预测。

- **端点**:
  - `POST /result/pack-result-analysis` - 测试工步结果分析
  - `POST /result/pack-result-predict` - 基于ML模型的电压预测
- **支持的模型架构**:
  - CatBoost
  - DeepSet
  - Transformer
- **功能**: 处理测试工步结果和电压数据，提供预测性分析

### 4. 工艺分析服务 (`/process`)

对制造工艺数据进行时序分析。

- **端点**:
  - `POST /process/pack-process-analysis` - 统计过程控制分析
  - `POST /process/pack-process-display` - 数据可视化处理
- **功能**:
  - 分钟级数据下采样
  - 百分位数带计算
  - DTW（动态时间规整）相似度计算

## 技术栈

| 类别 | 技术 |
|------|------|
| **Web框架** | FastAPI 0.95.2 + Uvicorn 0.23.1 |
| **生产服务器** | Gunicorn 20.1.0 |
| **数据库** | MySQL + SQLAlchemy 2.0.42 + PyMySQL |
| **数据处理** | Pandas 2.3.2 + NumPy 2.2.6 |
| **机器学习** | PyTorch (可选), CatBoost, joblib |
| **容器化** | Docker + Docker Compose |
| **测试框架** | Unittest + FastAPI TestClient |
| **代码质量** | Flake8 |
| **CI/CD** | GitLab CI |

## 项目结构

```
PACK_AI/
├── main.py                           # FastAPI 应用入口
├── requirements.txt                  # Python 依赖
├── Dockerfile                        # Docker 镜像构建
├── docker-compose.yml                # Docker Compose 配置
│
├── services/                         # 核心分析服务
│   ├── factory.py                   # 服务工厂
│   ├── temp_corr_service/           # 温度相关性分析
│   ├── dcr_analysis_service/        # DCR 异常检测
│   ├── result_analysis_service/     # 结果分析与预测
│   └── process_analysis_service/    # 工艺时序分析
│
├── configs/                          # 配置文件
│   ├── db_config.py                 # 数据库配置
│   ├── kafka_config.py              # Kafka 配置
│   ├── log_config.py                # 日志配置
│   └── monitor_config.py            # 监控配置
│
├── connects/                         # 数据库和 Kafka 客户端
│   └── db_client.py                 # 数据库客户端
│
├── core/                            # 核心工具
│   ├── config.py                    # 配置管理
│   └── logging.py                   # 日志设置
│
├── utils/                           # 工具函数
│
└── tests/                           # 测试套件
```

## 快速开始

### 环境要求

- Python 3.11+
- MySQL 5.7+
- Docker (可选，用于容器化部署)

### 本地开发

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd PACK_AI
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境变量**
   创建 `.env` 文件并配置必要的环境变量：
   ```env
   # 数据库配置
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_NAME=pack_ai_db

   # 应用配置
   APP_NAME=PACK_AI
   TENANT=jz2_pack
   ```

4. **启动服务**
   ```bash
   # 开发模式（自动重载）
   uvicorn main:app --reload --port 8000

   # 或直接运行
   python main.py
   ```

5. **访问API文档**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker 部署

1. **构建镜像**
   ```bash
   docker build -t pack_ai:latest .
   ```

2. **使用 Docker Compose 启动**
   ```bash
   docker-compose up -d
   ```

3. **访问服务**
   - 服务将在 http://localhost:8001 上运行
   - API文档: http://localhost:8001/docs

## API 端点

| 服务 | 端点 | 方法 | 描述 |
|------|------|------|------|
| 温度分析 | `/temp/pack-temp-corr` | POST | 计算温度与能量/容量相关性 |
| DCR分析 | `/dcr/pack-dcr-analysis` | POST | DCR异常检测和参数分析 |
| 结果分析 | `/result/pack-result-analysis` | POST | 测试结果分析 |
| 结果预测 | `/result/pack-result-predict` | POST | 基于ML模型的电压预测 |
| 工艺分析 | `/process/pack-process-analysis` | POST | 统计过程控制分析 |
| 工艺展示 | `/process/pack-process-display` | POST | 数据可视化处理 |

## 配置说明

系统通过 Pydantic 模型进行灵活配置：

- **数据库配置**: 支持生产和测试环境分离
- **模型存储**: 可配置的模型目录
- **测试工步**: 多种测试配置（如 330阶梯充、0.5C满充满放等）
- **电池包配置**: 支持 102 电芯和 96 电芯电池包

### 支持的电池包类型

- 102 电芯电池包
- 96 电芯电池包

### 支持的测试工步

- 330阶梯充
- 0.5C满充满放
- 1C充放电
- 其他自定义工步

## 数据流

```
原始电池数据 (MySQL)
    ↓
各分析服务 (基于包编码/车辆编码查询)
    ↓
分析算法处理
    ↓
结果返回 (REST API)
    ↓
ML模型预测 (可选)
```

## 主要特性

1. **多租户支持**: 可配置不同租户（如 jz2_pack）
2. **灵活配置**: 支持多种电池包规格和测试工步
3. **实时分析**: 快速处理，毫秒级响应日志
4. **统计方法**: 皮尔逊相关性、Z-score 异常检测、DTW 相似度
5. **机器学习**: 多种模型架构用于电压预测
6. **过程控制**: 动态边界的统计过程控制

## 测试

运行测试套件：

```bash
# 运行所有测试
python -m unittest discover tests

# 运行特定测试
python -m unittest tests/test_service_name.py
```

## 日志

日志文件位置: `ai_pack_server.log`

系统记录详细的运行日志，包括：
- API 请求和响应
- 处理时间（毫秒级）
- 错误和异常信息

## 生产部署

生产环境使用 Gunicorn + Uvicorn workers：

```bash
gunicorn -k uvicorn.workers.UvicornWorker main:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --timeout 120
```

Docker 部署已包含上述配置。

## 许可证

[请添加许可证信息]

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

[请添加联系方式]
