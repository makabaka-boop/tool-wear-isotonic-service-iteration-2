# 刀具磨损保序校正服务（Tool-Wear Isotonic Correction）

磨削线上的刀具磨损读数按加工顺序应只增不减，但传感器噪声会制造局部回落。
本服务对一批按顺序排列的加权观测做**加权最小二乘保序回归**（weighted
isotonic regression），返回唯一的非递减校正曲线，保留全部原始点；
计算全程只使用整数与不可约分数，**不使用浮点数**。

## 算法

采用 PAVA（Pool Adjacent Violators Algorithm，合并相邻违反者）：

1. 每个观测初始自成一块，块均值为其 `reading`。
2. 自左向右扫描；一旦栈顶块均值 **≥** 新块均值（比较用交叉相乘
   `n1*d2 ? n2*d1`，无浮点），就把两块合并。
3. 合并块的 fitted 取块内加权均值
   `mean = Σ(w·reading) / Σ(w)`；
   块内平方误差为精确分数
   `error = Σ(w·reading²) − (Σ(w·reading))²/Σ(w)`。
4. 总误差由各块误差用 `fractions.Fraction` 精确通分累加后约分。

**并列处没有选择空间**：相邻块均值相等时也一律合并，因此算法返回所有
最优解中块数最少（最粗）的规范分块，相同输入必然得到相同分块与逐点
fitted。输出包含：

- 每个连续合并块的含端点 0 基起止索引 `start_index` / `end_index`；
- 每块加权均值 `mean`；
- 每个原始点（同序、同 id、一个不少）的逐点 `fitted`；
- 总误差 `total_error = Σ weight·(fitted−reading)²`。

`mean`、`fitted`、`total_error` 均为已约分的 `{numerator, denominator}`
分数，分母恒为正。

### 固定观测（锚点）校正

经人工复测的磨损点必须原值保留，不能被平滑曲线改写。`POST
/correct/anchored` 在 `/correct` 的基础上接受 `anchors` 字段，以唯一
id 指定 **1～12 个锚点**；锚点的 `fitted` 以**等式约束**恒等于其
`reading`（不是用极大有限权重近似固定），其余点仍在整条非递减曲线上
最小化加权平方误差。

锚点把序列切成独立段：首锚点左侧只有上界、相邻锚点之间同时有上下界、
末锚点右侧只有下界。链式全序加常数界时，有界解等于无界保序解逐点
截断到界内，因此段内复用 PAVA 再精确截断，全程仍只用整数与分数。

```json
{
  "observations": [
    {"id": "p1", "reading": 3, "weight": 1},
    {"id": "p2", "reading": 1, "weight": 1},
    {"id": "p3", "reading": 2, "weight": 1}
  ],
  "anchors": ["p2"]
}
```

除 `/correct` 的全部 422 规则外，`anchors` 为空或超过 12 个、锚点 id
重复、锚点 id 不存在于本批观测中，均返回 **422**。

响应结构同 `/correct`，逐点用 `fixed` 标出固定点；分块仍为整条曲线
上的最大连续等值段——等值跨越锚点时也合并为同一块：

```json
{
  "blocks": [
    {"start_index": 0, "end_index": 1,
     "mean": {"numerator": 1, "denominator": 1}},
    {"start_index": 2, "end_index": 2,
     "mean": {"numerator": 2, "denominator": 1}}
  ],
  "fitted": [
    {"id": "p1", "fitted": {"numerator": 1, "denominator": 1}, "fixed": false},
    {"id": "p2", "fitted": {"numerator": 1, "denominator": 1}, "fixed": true},
    {"id": "p3", "fitted": {"numerator": 2, "denominator": 1}, "fixed": false}
  ],
  "total_error": {"numerator": 4, "denominator": 1}
}
```

锚点读数按观测顺序必须非递减（等值允许）。若出现下降，等式约束与非
递减曲线不可兼得，返回 **409** 与最早冲突的一对相邻锚点 id，不发布
任何部分拟合：

```json
{
  "detail": {
    "code": "INFEASIBLE_ANCHORS",
    "conflict": ["p1", "p2"]
  }
}
```

## 接口

`POST /correct`，普通 JSON：

```json
{
  "observations": [
    {"id": "p1", "reading": 3, "weight": 1},
    {"id": "p2", "reading": 1, "weight": 1},
    {"id": "p3", "reading": 2, "weight": 1}
  ]
}
```

约束（任一不满足则整次请求返回 **422**）：

- 观测数 2 ～ 5000；
- `id`：非空、ASCII；同批内唯一；
- `reading`：整数，0 ≤ reading ≤ 10⁹（不接受浮点/布尔/字符串）；
- `weight`：整数，1 ≤ weight ≤ 10⁶；
- 观测对象与顶层均拒绝任何未知字段。

响应：

```json
{
  "blocks": [
    {"start_index": 0, "end_index": 2,
     "mean": {"numerator": 2, "denominator": 1}}
  ],
  "fitted": [
    {"id": "p1", "fitted": {"numerator": 2, "denominator": 1}},
    {"id": "p2", "fitted": {"numerator": 2, "denominator": 1}},
    {"id": "p3", "fitted": {"numerator": 2, "denominator": 1}}
  ],
  "total_error": {"numerator": 2, "denominator": 1}
}
```

另有 `GET /health` 返回 `{"status": "ok"}`。

## 运行

Docker Compose（Python 3.12 镜像）：

```bash
docker compose up --build api          # http://localhost:8000
curl localhost:8000/health
```

在**同一镜像**内运行测试（领域算法 + HTTP）：

```bash
docker compose --profile test run --rm tests
# 或直接覆盖命令：
docker compose run --rm api python -m pytest -v
```

本地开发：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.main:app --reload
```

## 布局

```
app/
  isotonic.py   # 纯整数/分数 PAVA 与锚点有界拟合（无 float）
  schemas.py    # pydantic 请求/响应模型与 422 校验
  main.py       # FastAPI 路由（/correct、/correct/anchored、/health）
tests/
  test_isotonic.py  # 领域算法测试
  test_anchored.py  # 固定观测校正测试
  test_http.py      # HTTP 测试
  oracle.py         # 枚举全部连续分区（2^(n-1) 种）的精确神谕
pytest.ini / Dockerfile / docker-compose.yml
```

## 测试要点

- **全分区枚举对拍**：n≤7 的短序列枚举全部 reading/weight 组合与全部
  2^(n−1) 个连续分区（神谕只接受块均值非递减的可行分区，用 Fraction
  精确计算误差），断言 PAVA 的分块与总误差一致；另对 n≤9 的随机序列
  枚举对拍。
- 结构不变量：块连续覆盖、逐点 fitted 与块均值一致、fitted 非递减、
  分数全部已约分且分母为正、块误差之和等于逐点误差之和。
- 场景：**全降序**（合并为一块）、**零读数**、**大权重/大读数精确
  分数**、并列合并（最粗规范分块）、5000 点规模、重复调用确定性。
- 无浮点保证：字节码常量扫描 + AST 检查（无 float 常量、无 `/`
  真除法、无 `float` 内建）。
- 固定观测校正：短序列上用**独立有理数候选枚举**核对最优性——枚举
  全部连续分区、块值取 clip(加权均值, 下界, 上界) 的全部可行候选
  （另有一个不做段分解、不用截断性质的整序列暴力神谕交叉核对段分解
  假设）；覆盖相等锚点、端点锚点、全固定、跨锚点等值单分块、锚点
  精确固定（非大权重近似）、空锚点与普通拟合逐项一致、最早冲突对
  报告，以及冲突后 `/correct` 与修正后的锚点请求仍可正常调用。
- HTTP：200 响应结构与精确数值、未知字段/重复 id/越界/类型错误/错误
  媒体类型/畸形 JSON 一律 422、未知或重复锚点 422、锚点下降 409
  （`INFEASIBLE_ANCHORS`）、2 与 5000 个点的边界。
