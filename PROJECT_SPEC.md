# 工程规格文档：面向交通流基本图的LLM物理规律理解能力评估

## 一、项目概述

### 1.1 研究目标

评估大语言模型（LLM）是否表现出对交通流基本图（Fundamental Diagram）物理规律的一致性。具体而言，通过设计三级递进的 Physics-Constrained Prompt，测试 LLM 在不同程度物理知识注入下，对速度-密度关系的预测表现。

### 1.2 核心研究问题（RQ）

- **RQ1**：LLM 是否具备基本图的定性规律认知（密度增加 → 速度下降）？
- **RQ2**：在 prompt 中注入不同程度的物理知识，能否提升 LLM 的预测准确性？
- **RQ3**：LLM 更适合做定性物理推理，还是定量参数标定？

### 1.3 评估指标

- **MAE**（Mean Absolute Error）：平均绝对误差
- **RMSE**（Root Mean Square Error）：均方根误差
- **MAPE**（Mean Absolute Percentage Error）：平均绝对百分比误差
- **可视化**：LLM 预测的 v-k 曲线叠加在真实散点图 + Greenshields 标定曲线上
- **单调一致率**（自然语言描述）：在相邻密度对中，LLM 给出正确速度递减方向的比例

### 1.4 预期最终交付物

1. 一个完整的 Python 项目，包含数据处理、Prompt 构建、API 调用、结果分析、可视化
2. 所有实验结果的 CSV 文件
3. 所有可视化图表（PNG 格式）
4. 一份论文用的结果汇总表格

---

## 二、项目结构

```
traffic_fd_llm/
├── README.md                   # 项目说明
├── requirements.txt            # 依赖
├── config.py                   # API keys、模型配置、文件路径等全局配置
├── data/
│   ├── raw/                    # 原始 PeMS08 数据（npz）
│   └── processed/              # 处理后的 CSV 数据
├── src/
│   ├── data_prepare.py         # 模块一：数据加载、密度计算、传感器选择
│   ├── fd_calibration.py       # 模块一：Greenshields 最小二乘法标定
│   ├── prompt_builder.py       # 模块二：三级 Prompt 构建器
│   ├── llm_caller.py           # 模块二：LLM API 调用与结果解析
│   ├── evaluation.py           # 模块四：指标计算（MAE, RMSE, MAPE）
│   └── visualization.py        # 模块四：绘图（基本图散点、预测曲线叠加等）
├── experiments/
│   ├── run_all.py              # 一键运行所有实验的主脚本
│   └── results/                # 实验结果输出目录（CSV + PNG）
└── paper/
    └── figures/                # 论文用图表输出目录
```

---

## 三、模块一：数据准备与基本图标定

### 3.1 数据来源

- **数据集**：PeMS08
- **下载方式**：`git clone https://github.com/divanoresia/Traffic.git`
- **文件**：`Traffic/PEMS08/PEMS08.npz`
- **数据格式**：numpy array，shape = `(17856, 170, 3)`
  - 维度 0：时间步（17856 个，每步 5 分钟，覆盖 2016.07.01 - 2016.08.31）
  - 维度 1：传感器（170 个）
  - 维度 2：特征（0=flow veh/5min, 1=occupancy 0~1, 2=speed mph）

### 3.2 数据处理步骤（`data_prepare.py`）

```
输入：PEMS08.npz
输出：processed/{sensor_id}_fd_data.csv，列为 [timestamp, flow_veh_h, speed_kmh, density_veh_km]
```

**具体处理：**

1. 加载 npz 文件，提取 data array
2. 对每个目标传感器：
   - `flow_veh_h = flow_raw * 12`（从 veh/5min 转为 veh/h）
   - `speed_kmh = speed_mph * 1.60934`（从 mph 转为 km/h）
   - `density_veh_km = flow_veh_h / speed_kmh`（基本方程 q = k * v）
   - **过滤条件**：剔除 `speed_kmh < 5` 的记录（避免除零和异常低速）；剔除 `flow_veh_h == 0` 的记录
3. 保存为 CSV

**传感器选择策略：**

需要从 170 个传感器中选出 3 个代表性传感器：

- **主传感器（混合状态）**：平均速度在 40-60 km/h 之间，且速度标准差较大（说明既有自由流也有拥堵）
- **补充传感器 A（自由流为主）**：平均速度 > 80 km/h
- **补充传感器 B（拥堵为主）**：平均速度 < 40 km/h

选择方法：遍历所有 170 个传感器，计算每个传感器的 speed 均值和标准差，然后按上述标准筛选，每类选 1 个。如果某类找不到，放宽阈值。

**输出示例（CSV 文件前几行）：**

```csv
timestamp,flow_veh_h,speed_kmh,density_veh_km
0,1596.0,105.9,15.07
1,1368.0,107.7,12.70
2,1680.0,107.5,15.63
```

### 3.3 Greenshields 模型标定（`fd_calibration.py`）

```
输入：processed/{sensor_id}_fd_data.csv
输出：标定参数 (v_f, k_j) 和拟合曲线数据
```

**Greenshields 模型**：`v = v_f * (1 - k / k_j)`

其中：
- `v_f`：自由流速度（km/h），即密度为零时的速度
- `k_j`：阻塞密度（veh/km），即速度为零时的密度

**标定方法**：使用 `scipy.optimize.curve_fit` 对散点数据 (density, speed) 进行最小二乘拟合。

```python
from scipy.optimize import curve_fit

def greenshields(k, v_f, k_j):
    return v_f * (1 - k / k_j)

# popt = [v_f, k_j]
popt, pcov = curve_fit(greenshields, density_data, speed_data, 
                        p0=[100, 150],           # 初始猜测
                        bounds=([50, 50], [150, 300]))  # 参数范围
```

**输出**：
- 打印并保存标定参数：`v_f = XX km/h, k_j = XX veh/km`
- 生成基本图散点 + 拟合曲线图（保存为 PNG）

---

## 四、模块二：三级 Prompt 设计与 LLM 调用

### 4.1 Prompt 构建器（`prompt_builder.py`）

构建三个级别的 prompt，每个 prompt 的结构为：

```
[系统背景（可选）] + [参考数据（可选）] + [任务指令] + [输出格式要求]
```

**所有 prompt 共用的输入数据格式：**

从标定用的散点数据中，随机抽取 8-12 个 (density, speed) 对作为参考样本，另外均匀选取 10 个密度值作为待预测点（覆盖从低密度到高密度的范围）。

**所有 prompt 共用的输出格式要求：**

```
请以 JSON 数组格式返回预测结果，每个元素包含 density 和 predicted_speed 两个字段。
只返回 JSON，不要包含任何其他文字。

示例格式：
[
  {"density": 10, "predicted_speed": 95.2},
  {"density": 20, "predicted_speed": 82.1}
]
```

---

#### Level 0: Naive Prompt（零物理知识）

```
以下是某高速公路检测器采集的交通数据。

参考数据（密度 veh/km → 实测速度 km/h）：
density=12.5, speed=98.3
density=25.1, speed=85.7
density=38.0, speed=71.2
density=51.3, speed=55.8
density=67.2, speed=42.1
density=80.5, speed=28.6
density=95.0, speed=18.3
density=110.3, speed=9.5

请根据以上参考数据，预测以下密度值对应的车速：
density = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

{输出格式要求}
```

**要点**：不提供任何交通流理论背景，只给原始数据，测试 LLM 的裸推理能力。

---

#### Level 1: Physics-Informed Prompt（自然语言物理概念）

```
你是一位交通工程专家。以下任务涉及交通流基本图（Fundamental Diagram）中的速度-密度关系。

背景知识：
- 交通流中，当道路上车辆密度增大时，车辆之间的间距缩小，驾驶员被迫降低速度
- 当密度很低（接近零）时，车辆可以自由行驶，速度达到最大值，称为"自由流速度"
- 当密度达到极限（完全堵死）时，车辆无法移动，速度降为零，此时的密度称为"阻塞密度"
- 速度随密度增加而单调递减

参考数据（密度 veh/km → 实测速度 km/h）：
{同 Level 0 的参考数据}

请根据以上交通流理论知识和参考数据，预测以下密度值对应的车速：
density = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

{输出格式要求}
```

**要点**：用自然语言描述交通流的物理规律，但不给出具体数学公式。

---

#### Level 2: Physics-Constrained Prompt（公式 + 参数范围 + 约束）

```
你是一位交通工程专家。以下任务涉及交通流基本图（Fundamental Diagram）中的速度-密度关系。

物理模型：
速度与密度的关系遵循 Greenshields 线性模型：
  v = v_f × (1 - k / k_j)
其中：
  v = 速度（km/h）
  k = 密度（veh/km）
  v_f = 自由流速度，根据该路段数据，大约在 {v_f_low} ~ {v_f_high} km/h 之间
  k_j = 阻塞密度，根据该路段数据，大约在 {k_j_low} ~ {k_j_high} veh/km 之间

物理约束：
  - 速度必须 >= 0
  - 速度必须 <= v_f
  - 密度越大，速度越低（严格单调递减）

参考数据（密度 veh/km → 实测速度 km/h）：
{同 Level 0 的参考数据}

请根据以上物理模型、约束条件和参考数据，预测以下密度值对应的车速：
density = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

{输出格式要求}
```

**要点**：
- `{v_f_low}` 和 `{v_f_high}` 取标定值的 ±15%（例如标定 v_f=105，则填 89~121）
- `{k_j_low}` 和 `{k_j_high}` 同理
- 直接给出 Greenshields 公式和物理约束

---

### 4.2 LLM API 调用（`llm_caller.py`）

```
输入：prompt 字符串
输出：解析后的 [(density, predicted_speed), ...] 列表
```

**支持的 LLM（均为免费/极低成本国产API）**：

所有接口均兼容 OpenAI SDK 格式，代码中统一使用 `openai` 库调用，只需切换 `base_url`、`api_key` 和 `model` 即可。

| 角色 | 模型 | 平台 | API base_url | model 参数 | 免费额度 |
|------|------|------|-------------|-----------|---------|
| 主模型 | DeepSeek-V3 | DeepSeek 官方 | `https://api.deepseek.com` | `deepseek-chat` | 注册送约500万tokens |
| 补充模型 | Qwen-Plus（通义千问） | 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` | 新用户每模型100万tokens |

> **备选方案**（如果上述平台不可用）：
> - 硅基流动 SiliconFlow：`https://api.siliconflow.cn/v1`，支持 DeepSeek-V3 等开源模型，注册送免费额度
> - 阿里云百炼也提供 DeepSeek-R1/V3 托管版本，model 参数为 `deepseek-v3` 或 `deepseek-r1`
>
> 以上平台全部兼容 OpenAI 接口格式，代码无需修改，只需在 config.py 中更换 base_url、api_key 和 model。

**注册流程（简要）**：
- DeepSeek 官方：访问 `platform.deepseek.com` → 手机号注册 → 创建 API Key → 即可使用
- 阿里云百炼：访问 `bailian.console.aliyun.com` → 阿里云账号登录 → 开通百炼服务 → 创建 API Key

**API 调用方式（统一使用 openai 库）**：

```python
from openai import OpenAI

# DeepSeek 示例
client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_tokens=1000
)

result_text = response.choices[0].message.content
```

```python
# 阿里云百炼（通义千问）示例
client = OpenAI(
    api_key="your-dashscope-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_tokens=1000
)

result_text = response.choices[0].message.content
```

**结果解析逻辑**：

1. 从 LLM 返回的文本中提取 JSON 数组（可能被包裹在 ```json ``` 代码块中）
2. 用正则表达式 `r'\[.*\]'`（dotall 模式）匹配 JSON 数组
3. 用 `json.loads()` 解析
4. 提取每个元素的 `density` 和 `predicted_speed` 字段
5. 如果解析失败，记录原始返回文本，标记为解析失败

**重试机制**：每次调用最多重试 3 次（应对网络超时或 API 限流）。

**稳定性处理**：对每个 prompt 配置运行 3 次，取预测结果的中位数作为最终值（减少 LLM 输出的随机性）。

---

## 五、模块三（分析用）：物理一致性检查

> 注意：这不是一个独立的方法模块，而是在分析结果时使用的辅助检查。

### 5.1 检查内容（在 `evaluation.py` 中实现）

**单调一致性检查**：

```python
def check_monotonicity(densities, speeds):
    """检查 speed 是否随 density 单调递减"""
    # 按 density 排序
    sorted_pairs = sorted(zip(densities, speeds))
    correct = 0
    total = 0
    for i in range(len(sorted_pairs) - 1):
        total += 1
        if sorted_pairs[i+1][1] <= sorted_pairs[i][1]:
            correct += 1
    return correct, total
    # 返回如 (8, 9) 表示"9对相邻密度中有8对满足单调递减"
```

**边界合理性检查**（仅用于文字描述，不作为正式指标）：

- 是否有负速度
- 是否有速度超过 v_f 标定值的 120%
- 是否有速度在高密度区域（如 k > 0.8 * k_j）仍然很高

---

## 六、模块四：评估与可视化

### 6.1 指标计算（`evaluation.py`）

```
输入：真实速度列表 y_true，预测速度列表 y_pred
输出：MAE, RMSE, MAPE 数值
```

```python
import numpy as np

def calc_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def calc_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 过滤 y_true 接近零的值，避免除零
    mask = y_true > 5  # 速度小于5km/h的不计入MAPE
    if mask.sum() == 0:
        return float('nan')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
```

**真实速度的获取方式**：

待预测的 10 个密度点不是随机的，而是从真实数据中选取的。对于每个待预测密度 k_i，在真实数据中找到最近的密度值对应的速度作为 y_true。具体做法：

1. 将真实散点数据按密度分成 10 个等宽区间（bin）
2. 每个 bin 的中心密度作为待预测密度 k_i
3. 每个 bin 内所有真实速度的均值作为 y_true_i
4. 将 k_i 发给 LLM 预测得到 y_pred_i

### 6.2 可视化（`visualization.py`）

需要生成以下图表：

#### 图 1：基本图散点 + Greenshields 标定曲线（数据概览）

- X 轴：density (veh/km)
- Y 轴：speed (km/h)
- 内容：真实数据散点（灰色半透明）+ Greenshields 拟合曲线（黑色实线）
- 标注：v_f 和 k_j 的标定值
- 保存为：`figures/fig1_fd_scatter_with_fit.png`

#### 图 2：LLM 三级 Prompt 预测对比（核心图）

- X 轴：density (veh/km)
- Y 轴：speed (km/h)
- 内容：
  - 真实数据散点（灰色半透明，背景）
  - Greenshields 标定曲线（黑色实线）
  - Level 0 预测点（蓝色圆圈 + 连线，虚线）
  - Level 1 预测点（橙色三角 + 连线，虚线）
  - Level 2 预测点（红色方块 + 连线，虚线）
- 图例：标明每条线的含义
- 保存为：`figures/fig2_llm_prediction_comparison.png`

#### 图 3：误差指标柱状图

- X 轴：方法（Greenshields LS, Level 0, Level 1, Level 2）
- Y 轴：分三个子图 — MAE, RMSE, MAPE
- 保存为：`figures/fig3_metrics_bar_chart.png`

#### 图 4（可选）：补充传感器验证

- 与图 2 相同格式，但使用补充传感器 A 和 B 的数据
- 保存为：`figures/fig4_supplementary_sensors.png`

#### 图 5（可选）：补充 LLM 对比

- 与图 2 相同格式，但叠加主 LLM 和补充 LLM 的 Level 2 预测结果
- 保存为：`figures/fig5_llm_comparison.png`

**绘图风格要求**：
- 使用 matplotlib，字体大小适中（标题 14pt，轴标签 12pt，图例 10pt）
- 中文标签使用 SimHei 或 WenQuanYi 字体（`plt.rcParams['font.sans-serif'] = ['SimHei']`）
- 如果中文字体不可用，使用英文标签
- DPI = 300
- 图片尺寸：单图 (8, 6)，子图 (12, 4)

---

## 七、主实验脚本（`run_all.py`）

这是一键运行所有实验的入口脚本，执行流程如下：

```
Step 1: 数据准备
  → 加载 PeMS08 数据
  → 选择 3 个传感器
  → 计算密度、保存 CSV
  → 对主传感器做 Greenshields 标定

Step 2: 构建 Prompt
  → 从主传感器数据中抽取参考样本和待预测密度点
  → 构建 Level 0 / 1 / 2 三个 prompt

Step 3: 调用 LLM
  → 对每个 prompt 调用主 LLM（3次取中位数）
  → 对 Level 2 prompt 调用补充 LLM（3次取中位数）
  → 解析并保存所有预测结果到 CSV

Step 4: 评估
  → 计算所有方法的 MAE, RMSE, MAPE
  → 检查单调一致性
  → 生成汇总表格（CSV）

Step 5: 可视化
  → 生成所有图表
  → 保存到 figures/ 目录

Step 6: 输出汇总
  → 打印所有结果到终端
  → 保存结果汇总到 results/summary.csv
```

**命令行用法**：

```bash
# 运行全部实验
python experiments/run_all.py

# 只运行数据准备（不调 API）
python experiments/run_all.py --step data_only

# 只运行评估和可视化（已有 LLM 结果时）
python experiments/run_all.py --step eval_only
```

---

## 八、配置文件（`config.py`）

```python
# ===== API 配置 =====
# 所有接口均兼容 OpenAI SDK，统一使用 openai 库调用

# --- 主模型：DeepSeek（官方平台，注册即送免费额度） ---
# 注册地址：https://platform.deepseek.com
DEEPSEEK_API_KEY = "your-deepseek-api-key-here"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# --- 补充模型：通义千问（阿里云百炼，新用户每模型100万tokens免费） ---
# 注册地址：https://bailian.console.aliyun.com
QWEN_API_KEY = "your-dashscope-api-key-here"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"

# --- 备选：硅基流动 SiliconFlow（注册送免费额度） ---
# 注册地址：https://cloud.siliconflow.cn
# SILICONFLOW_API_KEY = "your-siliconflow-api-key-here"
# SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
# SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V3"

# ===== 主/补充 LLM 角色分配 =====
PRIMARY_LLM = "deepseek"
SECONDARY_LLM = "qwen"

# ===== 实验参数 =====
NUM_REFERENCE_POINTS = 10   # 给 LLM 的参考样本数
NUM_PREDICT_POINTS = 10     # 待预测密度点数
LLM_REPEAT_TIMES = 3        # 每个 prompt 重复调用次数
LLM_TEMPERATURE = 0.1       # 生成温度

# ===== 数据路径 =====
RAW_DATA_PATH = "data/raw/PEMS08.npz"
PROCESSED_DATA_DIR = "data/processed/"
RESULTS_DIR = "experiments/results/"
FIGURES_DIR = "paper/figures/"
```

---

## 九、依赖（`requirements.txt`）

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
openai>=1.0
```

---

## 十、注意事项与常见问题

### 10.1 密度计算的单位问题

PeMS08 原始数据中：
- flow 是 **veh/5min**，必须乘以 12 转为 veh/h
- speed 是 **mph**，必须乘以 1.60934 转为 km/h
- density = flow_veh_h / speed_kmh，单位为 veh/km

如果不做单位转换，标定出的参数会完全错误。

### 10.2 LLM 返回格式不稳定

LLM 有时会在 JSON 前后加上解释文字或 markdown 代码块。解析时需要：
1. 先尝试直接 `json.loads(response_text)`
2. 失败则用正则 `re.search(r'\[.*\]', response_text, re.DOTALL)` 提取
3. 再失败则去掉 ```json 和 ``` 标记后重试
4. 如果仍然失败，记录原始文本，人工检查

### 10.3 MAPE 的除零问题

当真实速度接近 0 时（严重拥堵），MAPE 会趋向无穷大。解决方案：只对 speed > 5 km/h 的数据点计算 MAPE。

### 10.4 API 调用成本估算

每个 prompt 大约 300-500 tokens 输入 + 200-300 tokens 输出。
3 个 Level × 3 次重复 × 2 个 LLM = 18 次调用。
加上补充传感器实验，总计约 30-40 次 API 调用，消耗不到 10 万 tokens。

- DeepSeek 官方：注册送免费额度（约500万tokens），本实验消耗量可忽略不计
- 阿里云百炼：新用户每模型100万tokens免费，本实验消耗量可忽略不计
- **总成本：0 元**

### 10.5 如果实验结果中出现物理不一致

如果 LLM 输出有个别不合理的值（如负速度、非单调），在论文分析中指出这个现象即可。可以简单说明"对异常值做边界裁剪后指标变化如何"，但不要把它包装成一个独立的方法模块。

实现参考（仅用于分析讨论，不是核心方法）：

```python
def clip_speed(speeds, v_f):
    """将速度裁剪到 [0, v_f] 范围"""
    return [max(0, min(s, v_f)) for s in speeds]
```
