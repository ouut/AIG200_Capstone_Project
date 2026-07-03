# GCP Cloud Run 部署指南

## 架构概览

```
用户 → Cloud Run (FastAPI) → Artifact Registry (Docker 镜像)
                                ↓
                          本地构建: train.py → model.pt → Docker build → push
```

---

## 前提条件 (Prerequisites)

### 1. GCP 账号
- 在 [console.cloud.google.com](https://console.cloud.google.com) 注册
- 新用户有 **$300 免费额度**（90天），足够完成此作业
- Cloud Run 每月有 **200万次免费请求**

### 2. 安装 gcloud CLI

```bash
# macOS
brew install --cask google-cloud-sdk

# 或下载安装包: https://cloud.google.com/sdk/docs/install

# 验证安装
gcloud version
```

### 3. 初始化并登录

```bash
# 登录 GCP
gcloud auth login

# 设置默认项目（替换 PROJECT_ID 为你的项目 ID）
gcloud auth application-default login
```

---

## Step 1: 创建 GCP 项目

```bash
# 创建项目
gcloud projects create fashion-mnist-api-2026 \
    --name="FashionMNIST API" \
    --set-as-default

# 如果已有项目，直接设置
gcloud config set project YOUR_PROJECT_ID

# 确认当前项目
gcloud config get-value project
```

> **注意**：项目 ID 必须全局唯一，建议加上你的名字缩写，如 `fashion-mnist-api-<your-name>`。

---

## Step 2: 启用需要的 API

```bash
# 启用 Cloud Run API
gcloud services enable run.googleapis.com

# 启用 Artifact Registry API (存储 Docker 镜像)
gcloud services enable artifactregistry.googleapis.com

# 启用 Cloud Build API (可选，用于云端构建)
gcloud services enable cloudbuild.googleapis.com

# 查看已启用的 API
gcloud services list --enabled
```

---

## Step 3: 创建 Artifact Registry 仓库

```bash
# 创建 Docker 镜像仓库
# 选择离你最近的区域: asia-east1 (台湾), us-central1 (美国), europe-west1 (欧洲)
gcloud artifacts repositories create fashion-mnist-repo \
    --repository-format=docker \
    --location=asia-east1 \
    --description="FashionMNIST Docker images"

# 验证仓库创建成功
gcloud artifacts repositories list

# 配置 Docker 认证（让 docker push 能推到 GCP）
gcloud auth configure-docker asia-east1-docker.pkg.dev
```

> **常用区域代码**：
> - `asia-east1` — 台湾（亚太用户推荐）
> - `asia-southeast1` — 新加坡
> - `us-central1` — 美国中部
> - `europe-west1` — 比利时

---

## Step 4: 训练模型（本地）

在推送镜像之前，需要先训练并导出模型：

```bash
# 回到项目目录
cd /path/to/MachineLearningDeploymentAssignment_1

# 安装训练依赖
pip install torch torchvision numpy matplotlib seaborn scikit-learn joblib

# 运行训练脚本（会下载 FashionMNIST 数据并训练）
python train.py

# 训练完成后，确认模型文件存在
ls -lh models/
# 应该看到:
#   best_model.pt          ← TorchScript 模型
#   best_model_state.pt    ← 模型 state dict
#   preprocessor.joblib    ← 预处理 pipeline
#   metadata.json          ← 模型元数据
```

---

## Step 5: 构建 Docker 镜像

```bash
# 构建镜像
docker build -t fashion-mnist-api .

# 本地测试
docker run -p 8080:8080 \
    -e API_KEY="your-secret-api-key" \
    fashion-mnist-api

# 在另一个终端测试
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
    -H "X-API-Key: your-secret-api-key" \
    -H "Content-Type: application/json" \
    -d '{"image": ['$(python -c "import numpy as np; print(','.join(map(str, np.random.randint(0,256,784))))")']}'

# 如果一切正常，Ctrl+C 停止容器
```

---

## Step 6: 推送镜像到 Artifact Registry

```bash
# 获取项目 ID
PROJECT_ID=$(gcloud config get-value project)

# 给镜像打标签（注意替换区域）
docker tag fashion-mnist-api \
    asia-east1-docker.pkg.dev/$PROJECT_ID/fashion-mnist-repo/fashion-mnist-api:v1

# 推送到 GCP
docker push asia-east1-docker.pkg.dev/$PROJECT_ID/fashion-mnist-repo/fashion-mnist-api:v1

# 在 Console 中验证
# 打开: https://console.cloud.google.com/artifacts
```

> 如果命令没有反应，确认 `gcloud auth configure-docker <区域>-docker.pkg.dev` 已经执行过。

---

## Step 7: 部署到 Cloud Run

```bash
PROJECT_ID=$(gcloud config get-value project)
REGION="asia-east1"  # 和 Artifact Registry 同区域
IMAGE="asia-east1-docker.pkg.dev/$PROJECT_ID/fashion-mnist-repo/fashion-mnist-api:v1"

gcloud run deploy fashion-mnist-api \
    --image=$IMAGE \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=3 \
    --concurrency=80 \
    --timeout=60 \
    --set-env-vars="API_KEY=your-secret-api-key-here" \
    --port=8080
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--platform=managed` | 全托管 Cloud Run（无需管理 K8s） |
| `--allow-unauthenticated` | 允许公开访问（API key 由应用层验证） |
| `--memory=1Gi` | 1GB 内存（PyTorch 模型需要） |
| `--cpu=1` | 1 vCPU |
| `--min-instances=0` | 无请求时缩容到 0（省钱） |
| `--max-instances=3` | 最多 3 个实例 |
| `--concurrency=80` | 每个实例最多 80 个并发请求 |
| `--timeout=60` | 请求超时 60 秒 |
| `--port=8080` | 容器监听端口 |

### 部署成功后你会看到：

```
✓ Deploying new service...
✓ Creating Revision...
✓ Routing traffic...
Done.
Service URL: https://fashion-mnist-api-xxxxx-asia-east1.run.app
```

**记下这个 URL！这就是你的 API 端点。**

---

## Step 8: 测试部署的 API

```bash
# 替换为你的实际 URL
API_URL="https://fashion-mnist-api-xxxxx-asia-east1.run.app"
API_KEY="your-secret-api-key-here"

# 1. Health check
curl $API_URL/health

# 2. 发送预测请求
curl -X POST $API_URL/predict \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"image": ['$(python3 -c "
import numpy as np
img = np.random.randint(0, 256, 784).tolist()
print(','.join(map(str, img)))
")']}'

# 3. 使用 Python 测试
python3 test_api.py
# （需要修改 test_api.py 中的 BASE_URL 和 API_KEY）
```

---

## 常见问题

### Q1: `gcloud` 命令找不到
```bash
# 检查是否已安装
which gcloud

# 重新初始化
gcloud init
```

### Q2: 部署时报权限错误
```bash
# 确保已登录并有项目权限
gcloud auth list
gcloud projects list

# 设置 billing account（免费层也需要绑定）
# 在 Console: https://console.cloud.google.com/billing
```

### Q3: Cloud Run 启动失败 (Cold Start)
- 查看日志：`gcloud run logs read fashion-mnist-api --region=asia-east1`
- 常见原因：内存不足 → 尝试 `--memory=2Gi`
- 模型加载慢 → 增加 `--timeout=120`

### Q4: Docker 镜像太大
- PyTorch CPU 版本约 150MB
- 如果镜像超过 500MB，检查是否误包含 CUDA 版本
- 使用 `.dockerignore` 排除不需要的文件

### Q5: 想更新模型
```bash
# 1. 重新训练
python train.py
# 2. 重新构建
docker build -t fashion-mnist-api .
# 3. 推送新版本
docker tag fashion-mnist-api asia-east1-docker.pkg.dev/$PROJECT_ID/fashion-mnist-repo/fashion-mnist-api:v2
docker push asia-east1-docker.pkg.dev/$PROJECT_ID/fashion-mnist-repo/fashion-mnist-api:v2
# 4. 更新 Cloud Run
gcloud run deploy fashion-mnist-api \
    --image=asia-east1-docker.pkg.dev/$PROJECT_ID/fashion-mnist-repo/fashion-mnist-api:v2 \
    --region=asia-east1
```

---

## 费用估算

Cloud Run 免费额度（每月）：
- 200 万次请求
- 36 万 vCPU-秒
- 72 万 GiB-秒

对于本作业（假设每天 1000 次请求，每次 100ms）：
- ≈ 3000 次/月 → **完全免费**

---

## 交付清单

完成部署后，你需要提交：

- ✅ **API URL**: `https://fashion-mnist-api-xxxxx.run.app`
- ✅ **API Key**: 你设置的 key
- ✅ **代码仓库**: GitHub 链接
- ✅ **截图**: Cloud Run Console、API 测试结果
