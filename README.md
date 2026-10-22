# Polymarket-Alternative-Data

研究 Polymarket 信息能否为 A 股收益预测提供额外因子。包含历史沪深300 Qlib Alpha158/MLP baseline，以及原油预测市场采集、市场级因子和股票敏感度模型。所有输出均为预测研究，不包含交易执行。

## 实验口径

- 股票池：BaoStock 按指定历史日期返回的沪深300成分股，使用 **T 日成员**形成 T 日截面。来源按周更新，保存原始 `updateDate`，不将其视为精确到日的官方调整记录。
- 行情：默认 AKShare 东方财富日线，后复权；显式开启 `download.allow_sina_fallback` 后，失败请求可回退到 AKShare 新浪端点。下载研究期间所有历史成员的完整可用历史，包括研究起点前用于特征预热的行情。
- 特征与模型：Qlib Alpha158（158列）和 `baseline.model.MLPModel`，通过 Qlib `Model` 接口训练和预测。
- 目标：T 日收盘后预测 `close(T+2) / close(T+1) - 1`，T+1、T+2 均按全市场交易日历计算。本文以项目级设置和后续确认的一日目标为准。
- 标准化：在每个 T 日有效成员标签中使用普通 zscore，样本标准差 `ddof=1`；无效标签不进入训练，零方差截面不计分。
- 指标：逐日计算预测与标准化真实收益的 Pearson IC，按有效交易日等权汇总；同时提供 Rank IC。收益标签的 zscore 不改变 Pearson 相关系数。

项目覆盖2008-01-01至2026-09-30，默认测试2026年4月至9月。每月使用从2008年起、测试月开始前所有标签已知的历史样本，重新估计特征归一化参数、初始化模型和优化器，从头训练一次。训练轮数由 `model.epochs` 手动指定，当前 baseline 配置为20轮；省略该参数时回退为10轮。Baseline 和 Polymarket 扩展统一使用这一训练口径，不划分验证集，不使用早停或自动选 epoch。

| 测试区间 | 训练标签可用截止日 |
| --- | --- |
| 2026年4月 | 2026-03-31 |
| 2026年5月 | 2026-04-30 |
| 2026年6月 | 2026-05-31 |
| 2026年7月 | 2026-06-30 |
| 2026年8月 | 2026-07-31 |
| 2026年9月 | 2026-08-31 |

表中截止日指收益标签已可观察的日期。训练只排除测试开始时 T+2 标签尚不可知的样本。例如7月模型的最后一个训练特征日是2026-06-26，其标签使用6月29日至30日的收益。各测试月保留月末预测，事后计算标签；最终只评价 T+2 不晚于2026-09-30的样本。9月29日和30日仍输出预测，标签为 `NaN`，不计入 IC。所有训练样本等权。

## 安装和运行

使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)，在仓库根目录运行：

```bash
uv sync --locked --group dev
uv run python -m baseline all --config configs/baseline.yaml
```

也可以分别运行各阶段：

```bash
uv run python -m baseline download
uv run python -m baseline prepare
uv run python -m baseline train
uv run pytest -q
```

配置集中在 `configs/baseline.yaml`，相对路径以当前工作目录为基准。`train_start` 设置固定的训练起点，`first_test_month` 设置首个测试月，之后按月扩展训练历史。`download_end: null` 表示下载截至运行日期的可用行情；研究与评价边界仍固定为 `research_end`。`--refresh` 强制重建所选阶段产物。下载和特征准备复用完整且匹配的缓存；每次 CLI 训练新建独立运行目录并重新训练。网络下载中断后可以继续运行同一命令。下载失败会记录问题并终止，避免把缺失股票静默当作有效样本池。

若 BaoStock 返回 `10001011`（黑名单用户），下载会保留检查点并立即停止后续请求，不再重试或重新登录；待服务恢复访问后，运行同一命令即可续传。

默认 MLP 为 `158 → 32 → 1`，隐藏层使用 Tanh；Adam 学习率0.001、权重衰减0.0001、batch size 512。每月按 `model.epochs` 训练固定轮数，使用最后一轮权重预测；特征均值与标准差只用该月完整训练历史估计，缺失特征在标准化后置零。随机种子42，PyTorch使用2个CPU线程，Qlib特征计算使用1个进程，六个测试月串行执行。BaoStock 历史成员查询默认使用 `download.membership_workers: 1`，保持单连接串行，避免并发触发 IP 封禁。依赖解析锁定在 `uv.lock`。首次全量下载需要较长时间；年度特征缓存避免六轮重复计算 Alpha158。

## 数据和产物

```text
akshare/hs300/
  calendar.txt                 # 完整市场交易日，每行一个 YYYY-MM-DD
  instruments/YYYY-MM-DD.txt   # 当日成员，每行一个 SH600000/SZ000001
  data/SH600000.csv            # AKShare 后复权行情及 raw_close
  metadata.json               # 下载区间、成员来源更新日期
data/qlib/hs300/               # Qlib 日历、成员有效区间及二进制行情
data/alpha158/                 # 按年缓存特征、标准化标签及原始标签
outputs/baseline/
  YYYYMMDDTHHMMSSffffff/        # 每次 CLI train / all 运行的北京时间戳
    training_config.yaml       # 本次运行的完整配置快照
    training_config.json
    YYYY-MM/                  # 月度模型、预测、IC 和实验记录
    predictions.parquet / daily_ic.csv / monthly_ic.csv / summary.json
```

行情CSV保留日期、股票代码、开高低收、成交量、成交额、振幅、涨跌幅、涨跌额、换手率，另加未复权 `raw_close`。原始成交量单位为手、成交额单位为元。转换到 Qlib 时，令 `factor=后复权收盘价/raw_close`，成交量为 `原始成交量×100/factor`，VWAP为 `成交额/(原始成交量×100)×factor`，与后复权价格保持一致。

`download.allow_sina_fallback` 默认关闭。显式开启后，东方财富未复权请求重试失败时，仅从 AKShare 1.19.1 自带的新浪原始历史端点和解码器提取 `raw_close`，按后复权日期严格对齐，缺少目标日期仍报错；后复权请求重试失败时，改取新浪原始 OHLCV、成交额和 `hfq.js`，按 SDK 公式将原始 OHLC 乘当日有效复权因子并保留两位小数。只向后延续已生效的因子，绝不填充价格或增加原始记录中不存在的日期，原始停牌空档保持缺失；重复日期、无效收盘价及缺失或无效因子仍报错。新浪成交量由股除以100转为手，成交额保持元；振幅、涨跌幅和涨跌额由请求区间内相邻已观察到的后复权日线推导，首行缺失；换手率保留缺失，不额外请求股本。全部请求使用配置的超时。新元数据记录 `price_source`、`raw_close_source`、回退原因及完整回退时的复权方法、单位与衍生字段；已有完整 CSV 和旧元数据继续复用。

Qlib provider 保留研究起点前的行情，但在研究结束日截断，阻止最终标签越界。停牌、零成交量和缺失行情保持缺失，不填充价格或压缩交易日历。成员退出后再次加入会保留为多个有效区间。

整个成员窗口都没有有效行情时，仅在真实有效报价能从前后包围全部成员日、窗口内没有异常 OHLC 等坏值时保留该股票及原始缺口；provider 的 `zero_member_bars` 和日志记录这些情况，仅含窗口前历史的截断输入仍会报错。

部分股票的接口后复权历史含有非正或非有限 OHLC。原始 CSV 保留这些记录，每股下载元数据的 `adjusted_ohlc_anomalies` 记录异常数量、日期和研究期内数量；转换到 Qlib 时对应交易日的行情设为缺失，依赖异常收盘价的收益标签也保持缺失。整只股票没有有效后复权行情、未复权收盘价无效或量额字段无效时，下载仍会失败。

年度缓存采用每个真实交易日完整历史成员的 `(datetime, instrument)` 索引，分别保存158维特征、标准化标签及原始标签。成员日超过该股最后一条行情时，仍保留全 `NaN` 行，不填充行情；未知标签样本不参与训练，测试预测仍保留。缓存指纹包含 `membership-index-v1`，manifest 记录 `index_policy` 与 `index_version`，旧索引缓存需重新生成。每月装载从 `train_start` 到当轮训练截止日的完整历史，以及当月测试数据，使用 `DataHandlerLP.from_df` 和 `DatasetH` 组织数据；复用年度特征，不重复计算 Alpha158。

Baseline 的 `output_dir` 是运行结果的父目录。CLI 执行 `train` 或 `all` 时，新建精确到微秒的北京时间目录，并在开始运行阶段前保存解析后的 `training_config.yaml` 和 `training_config.json`，包含绝对数据路径、本次输出路径、研究区间及模型参数。直接调用 `baseline.train.train(config)` 时仍使用传入的具体目录，可续跑该目录中的已完成月份。

`uv run python -m baseline.quintiles` 默认读取最近一次已完成运行的预测，将五分组分析保存到该运行的 `quintiles/` 子目录；也可用 `--predictions` 和 `--output-dir` 指定路径。

每次运行目录下 `YYYY-MM/` 保存 `model.pkl`（含归一化参数的完整 Qlib 模型）、`model.pt`（网络权重）、`training_history.csv`、`predictions.parquet`、`daily_ic.csv`、`metrics.json`、`run.json` 和完整配置；`completed.json` 用于校验续跑产物。`run.json` 的 `window`、`segments`、`actual`、`model` 描述训练范围和参数，`trained_epochs` 记录实际训练轮数，训练方案为 `expanding_monthly_fixed_epochs`。不生成验证指标或选轮数阶段产物。运行目录顶层 `predictions.parquet`、`daily_ic.csv`、`monthly_ic.csv`、`summary.json` 汇总六轮结果。预测列为 `score`、`label`（zscore标签）、`raw_label`（原始收益）。Qlib实验记录保存在本次运行的 `mlflow.db`，对应文件位于 `artifacts/`。

固定轮数训练的模型指纹已升级，旧模型需要重训。研究结束日由9月18日延长至9月30日后，先补齐原始行情和历史成分，再重新生成覆盖新区间的 provider 和 Alpha158 缓存：

```bash
uv run python -m baseline download
uv run python -m baseline prepare --refresh
uv run python -m baseline train --refresh
```

普通下载会复用已完成的原始记录；旧 provider 和 Alpha158 缓存不能直接覆盖新增研究区间。

## 验证与数据限制

离线测试覆盖收益对齐、截面标准化、股票池退出重入、停牌、复权量额、训练历史逐月扩展、固定轮数训练、T+2 标签边界和完整 Qlib 训练链路。测试中的合成行情仅用于验证程序，不作为研究结果。

BaoStock 的历史成分股来自周度快照，可能无法反映周内临时调整的精确生效日。AKShare 公共行情源可能存在限流、退市股票历史缺失或历史数据修订；缓存元数据和失败报告用于核查实际覆盖范围，完整性检查通过后才允许训练。

接口参考：[BaoStock](https://github.com/zxygithub/baostock/blob/master/docs/pythonAPI.md)、[AKShare](https://akshare.akfamily.xyz/data/stock/stock.html)、[Qlib Alpha158](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/contrib/data/handler.py)、[Qlib Model 接口](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/model/base.py)。

## Polymarket 扩展

配置为 `configs/polymarket.yaml`，包名 `polymarket_ext`；官方 SDK `polymarket-client` 占用 `polymarket` 导入名。SDK 固定为 0.13.0，PyTorch 默认安装 CPU 版本。采集只使用公开行情接口，无需账户、私钥或付费语言模型。

先准备 A 股日历和 Alpha158 缓存，再运行扩展：

```bash
uv sync --locked --group dev
uv run python -m baseline download
uv run python -m baseline prepare
uv run python -m polymarket_ext all --config configs/polymarket.yaml
```

可单独运行各阶段。仅研究市场级因子时运行到 `prepare` 即可，不需要 Alpha158 缓存，但必须提供真实 A 股交易日历：

```bash
uv run python -m polymarket_ext discover
uv run python -m polymarket_ext classify
uv run python -m polymarket_ext download
uv run python -m polymarket_ext prepare
uv run python -m polymarket_ext train
```

`discover` 分页遍历已关闭及未关闭事件，保存完整 Gamma 响应。默认 `collection.event_tag_slug: oil` 获取官方 oil 标签目录，再由分类规则筛选五类 WTI 市场；标签目录不保证涵盖全站所有原油问题，需要全站发现时设为 `null`。更改标签会自动重建发现缓存。若已有项目中提及的事件集，可将 `collection.events_file` 设为 Gamma 原始事件 JSON 数组或 JSONL 路径，每个 event 包含 `markets`。600 个 event / 3,340 个 market 是研究时的统计，不作为今天目录的固定数量。下载不以研究终点截断；因子与评价范围由 `research_start/end` 限定。

普通重跑复用下载快照并恢复未完成分页；使用 `--refresh` 更新已缓存的在线目录、行情及交易。网络失败会记录状态并使下载命令失败。`api_complete` 只表示请求与分页成功，不代表官方保证全生命周期历史完整。

默认按 market 使用 16 个下载线程，每个线程独立持有官方 SDK 客户端，历史价格按最长 15 天的窗口请求。可通过 `collection.workers` 调低并发；未配置时为串行。`data/polymarket/collection.json` 持续记录 `processed/total`、`completed` 和 `failed_markets`，每个 market 的 `coverage.json` 记录两侧报价及成交覆盖。只下载数据时依次运行 `discover`、`classify`、`download`，无需准备 A 股数据或启动训练。

事件分页分别记录来源身份；切换标签后即使中途失败，也不会复用其他标签的旧页。market 的原始响应单独绑定 token 等来源身份，来源变化时将旧 `raw` 原子移至该 market 的 `raw_archive`，新来源已成功的分页可继续复用，避免中断后混入旧 token 的价格或成交。

### 官方数据能力

| 数据 | 接口与采集方式 | 实际限制 |
| --- | --- | --- |
| event / market / outcome token | Gamma `/events/keyset`，官方 `PublicClient.list_events` | 同时读取 open / closed；保留原始规则、时间与 token 名称 |
| Yes/Up 与 No/Down 历史价格 | CLOB `/prices-history` 请求 `fidelity=1`；官方 SDK 的 Data v2 `list_price_history` 补充不同粒度 | 逐时间窗口请求最细数据；记录每条数据的实际粒度，不从 Yes 推算 No |
| 逐笔交易 | Data `/v2/trades`，官方 `PublicClient.list_trades`，cursor 分页 | 按市场查询为最近三年的滚动窗口；最小数量 0.01 token；保存逐笔原始价格、数量、方向和时间 |

Data v2 文档给出的最低保存期限为：1 分钟 7 天、5 分钟 60 天、30 分钟 90 天；3 小时历史可追溯至 2022-11-18。程序也请求 CLOB 的一分钟历史，保留它仍能提供的较细报价，再按实际响应补充较粗数据；不会把请求一分钟等同于获得连续一分钟历史。[Data v2 官方 OpenAPI](https://data-api.polymarket.com/v2/openapi.json) 将聚合点的 `timestamp` 定义为桶起点，价格观察发生在该桶内，因此只有 `timestamp + resolution_seconds` 不晚于截面时才可使用。`resolution_seconds=0` 表示精确 tick（包括返回窗口的末次真实观察），按原时间戳保留，不一概视作合成结算点。交易使用 taker 视角避免把一笔撮合按双方重复计算，也不基于相同时间和价格误删独立成交。

官方资料：[统一 Python SDK](https://github.com/Polymarket/py-sdk)、[CLOB 历史价格](https://docs.polymarket.com/api-reference/markets/get-prices-history)、[Data v2 历史价格](https://docs.polymarket.com/api-reference/markets/get-a-tokens-price-history)、[Data v2 逐笔交易](https://docs.polymarket.com/api-reference/feeds/list-trades)、[Gamma 事件分页](https://docs.polymarket.com/api-reference/events/list-events-keyset-pagination)。CLOB 历史接口尚无统一 SDK 的专门方法，因此通过该 SDK 的 HTTP transport 调用；与固定 SDK 版本相关的适配集中在 `collect.OfficialClient`。

### 分类与落盘

| 目录类别 | 含义 |
| --- | --- |
| `direction` | 区间 Up / Down，统一日度和多日 |
| `threshold` | 收盘或结算超过门槛 |
| `touch` | 期间上触 / 下触，各自分组 |
| `range` | 结算价格区间，含开放尾部 |
| `record` | 截止指定时间创历史新高 |

```text
data/polymarket/
  raw/events.jsonl
  raw/discovery.json
  raw/event_pages/
  markets.jsonl                       # 标准化的 market 目录
  classification.json                 # 统计、来源指纹和元数据假设
  classification_review.json          # 未识别、歧义、缺关键字段或明确排除的条目
  collection.json                     # 采集完成/失败状态
  <category>/<event-slug>--<id>/
    meta.json                         # event 原始信息
    <market-slug>--<id>/
      meta.json                       # 原始规则 + 标准化字段
      yes_prices.csv                  # Up 或 Yes，按 outcome 名称定位
      no_prices.csv                   # Down 或 No，独立下载
      trades.jsonl
      coverage.json                   # 分辨率、实际起止、空洞与接口限制
      raw/                            # 价格窗口和交易分页原始响应
data/polymarket_factors/
  aligned.parquet
  factors.parquet
  manifest.json
outputs/polymarket/
  YYYYMMDDTHHMMSSffffff/          # 每次 CLI train / all 运行的上海时间戳
    YYYY-MM/
      model.pkl / sensitivities.csv / training_history.csv
      predictions.parquet / daily_ic.csv / run.json
    daily_ic.csv / monthly_ic.csv / predictions.parquet / summary.json
```

Polymarket 的 `output_dir` 是运行结果的父目录。CLI 执行 `train` 或 `all` 时，会在其中新建一个精确到微秒的北京时间目录，避免不同运行覆盖彼此；该目录纳入 Git 跟踪。

规则分类排除汽油、Brent 等非目标市场；不根据数组位置猜测 Yes/Up token。收盘阈值必须有收盘或结算语义；无法判断触价方向、多重障碍或缺纪录门槛的市场进入复核清单。可复制 `configs/polymarket_overrides.example.yaml`，按真实 market ID 补充类别、价格口径、原始观察窗口和门槛，然后设置 `classification.overrides_file` 并重跑 `classify`。开放尾部写成字符串 `'-inf'` / `'inf'`，以保持标准 JSON。修订规则元数据时需依原始合约核实，不能填入事后结果。

`direction` 同时纳入日度与多日涨跌，沿用同一类别、编码器和股票敏感度；暂不扩展到日内涨跌。标题包含具体钟点不能单独作为日内排除条件，应按规则实际比较的价格时刻判断。时间无法确定的市场仍保留，待人工 overrides 补充；默认不调用语言模型。

涨跌比较时刻保存在 `observation_start/end`，可以逐项为空；平台到期字段 `expires_at` 独立保留。提取结果带有 `time_extraction_version: 1` 和 `time_extraction`，后者记录 `status`（`complete` / `partial` / `missing`）、两端依据 `start_evidence/end_evidence` 及 `issues`。不完整提取也会进入 `classification_review.json`，以 `included: true` 表明市场仍被纳入，供复核和补充 overrides。

自动提取覆盖的明确规则模板包括：带年份、时区的 `from/to` 比较，明确两日期的 `higher/lower than` 比较，以及 CME/NYMEX CL 规则中的前一交易日、同日开收盘、`first/last trading day of Month YYYY` 模板。交易日与开收盘采用固定版本 `pandas-market-calendars==5.4.0` 的 `CMEGlobex_CL` 日历。该包提供离线日历规则，不保证覆盖交易所实时发布的特殊调整；不明确的交易所或 settlement 时刻保持缺失，不用 close 时刻替代 settlement。特殊合约需核对原规则后通过 overrides 填写带时区的 `observation_start/end`。尚未接入语言模型补全服务。

Gamma 当前快照不提供完整的历史规则修订版本。程序保留 `createdAt/updatedAt`，以创建时间作为元数据最早可用时间，并明确记录“当前合约条款自创建以来未改变”的假设；仅凭今天的快照不能证明历史条款未被修改。其他类别未从规则中解析到观察起点时使用 event 级起点，特殊窗口应通过覆盖文件核实；涨跌类的比较起止价格时刻须来自规则，不能用 event 或 market 的创建、开放或实际结算时间替代。该限制独立于报价对齐，不能据此宣称完全消除历史元数据偏差。

### 因子计算口径

全部公式遵循 AGENTS.md。每个真实 A 股交易日以北京时间 15:00 为截面，只使用不晚于该时刻的报价；市场必须已开放且元数据可用，观察结束、到期或关闭后不再有效。默认超过 24 小时的报价视为缺失，可通过 `max_quote_age` 调整或设为 `null`。保留报价时间、年龄、原始价格、实际分辨率和有效性，不做向后填充或未来插值。

Data v2 聚合点等待其采样桶结束后才进入候选集；精确 tick 与 CLOB 点按返回的观察时间处理。在当时已可用的候选中选观察时间最新者，旧粗桶晚到不会覆盖较新的已知报价。面板额外保留 `quote_available_at` 和 `quote_source` 供核验。因子面板版本已升级为 3；旧版本需重建并重新训练。

区间涨跌类保留 `PROB`、`ENTROPY`、`DIRECTION_SCORE`、`DIRECTION_CERTAINTY`，并增加两个无量纲时间因子。设规则用于比较涨跌的起止价格时刻为 $T_s$、$T_e$，当前 A 股收盘截面为 $t$：

| 因子 | 计算式 |
| --- | --- |
| `WINDOW_LENGTH` | $\ln(1+(T_e-T_s)/1\text{天})$ |
| `TIME_TO_END` | $\ln(1+(T_e-t)/1\text{天})$ |

时间差按实际秒数除以 86400 计算，不按交易日计数。时刻从原始规则模板与对应交易所日历提取，处理规则时区与夏令时后转为 UTC。窗口长度要求两端有效且终点晚于起点；距终点时长只需有效终点且终点晚于当前截面，到达终点的市场已不再有效。缺少起点时只缺失窗口长度；无法确定终点时两个时间因子均缺失。时间缺失不删除市场，也不影响其他有效因子；缺报价时仍可计算这两个时间因子。

组别由 event、题型、标的、价格口径、观察窗口、到期时间以及触价方向共同确定。中心与尺度只使用该截面已开放成员的去重有限边界，即使某成员暂缺报价也不会把相邻门槛跨过去。相对位置至少要求两个价格水平；重复门槛使涉及它的相邻质量无效。显著负质量记为 NaN，容差内的小负值置零。

D1/D5 按完整 A 股交易日历在同一 token 内计算；D5 必须连续六个有效截面。开放尾部用反正切有界坐标计算宽度和密度，缺报价、非法边界或无效尺度保留 NaN。输出为市场级长表，保留每个市场的全部指定因子及分组中心、尺度。

`prepare` 逐 market 读取历史报价并对齐收盘截面，只在内存中汇总市场日表，随后统一计算跨 market 的分组因子。报价历史再多也不需要同时载入全部 token；分组中心、相邻门槛和 D1/D5 的计算口径保持一致。

### 股票敏感度模型与验证

扩展模型是 Qlib `Model` 子类，使用 `DatasetH`、`DataHandlerLP` 和 Qlib IC 接口。每只训练股票有五个可训练的 `b_i`，初始值为 1，允许负值。因子仅使用训练段的有限数值估计标准化参数，标准化后将缺失数值槽填为 0；编码器接收 `[标准化因子 × b_i, 逐因子 missing]`，其中 `missing=1` 表示 NaN 或正负无穷，`missing=0` 表示真实数值（包括真实的 0）。缺失标记不乘 `b_i`，不另设可训练的缺失向量。

当天各类活跃市场均保留，包括缺报价或全部因子缺失的真实市场行。无报价时仍可计算的位置、区间宽度等因子参与模型；其余因子逐项标为缺失。编码后的市场向量按类别取均值，附 `log(1 + 市场数量)` 和类别缺失标记，再与 Alpha158 拼接预测收益。类别缺失标记仅在该类全部真实市场均无有限因子时为 1，无市场时也为 1；至少一个因子有效则为 0。没有市场时均值向量和数量均为 0；用于凑齐批次长度的填充行不参与均值、数量或有效性判断。每个市场都在聚合前输入编码器，市场数量可以变化，新 market 不需要新增模型列。新股票的 `b_i` 使用中性值 1。

五个 `b_i` 与 MLP 预测网络联合反向传播训练并保存到 `sensitivities.csv`。Baseline 与扩展共享 MLP 预测头、训练流程和 Alpha158 归一化；扩展的输入增加五类市场编码、数量及类别缺失标记。参数是模型内部的相对敏感度，其尺度也受编码器影响，不宜当成已识别的经济因果暴露。

编码器只计算批次中的真实 market 行，再按股票样本聚合，避免为长期没有 Polymarket 数据的历史样本重复编码填充行。真实但因子全缺失的 market 仍参与编码；整批无市场时保留编码器参数的零梯度，使 Adam 动量和权重衰减与完整填充计算一致。

日度与多日涨跌共同使用 `direction` 的六个因子、一个类别编码器和每只股票的同一个 `b_i`。默认该编码器为 `12 → 32 → 32`，其他四类保持原结构；类别总数仍为五，最终 MLP 输入仍为 `158 + 5 × (32 + 2) = 328` 维。两个时间因子遵循相同的训练段标准化、敏感度缩放和逐因子缺失标记规则。

扩展从 `baseline_config` 继承历史股票池、普通截面 zscore 标签、月度窗口和 MLP 公共参数；`configs/polymarket.yaml` 默认只额外设置敏感度正则 `sensitivity_l2`。如显式覆盖扩展的 `model.epochs` 等参数，最终配置会记入 `run.json`。Baseline 与扩展均在每个测试月之前的完整可用历史上，从头训练手动指定的固定轮数；不划分验证集，使用最后一轮权重。两条流程均报告测试集逐日 Pearson IC 和 Rank IC，输出9月29日和30日的预测，但不评价超出研究终点的标签。

Polymarket 的可用历史远短于 2008 年以来的股票历史；2008 年等早期股票样本仍参与训练，始终保留五类输入，五个类别缺失标记均为 1。部分类型缺席、单项 D5 缺失和真实数值为 0 也有各自对应的标记。全缺失因子的数值槽对 `b_i` 不产生任务损失梯度；若某类在整个训练段从未有有效因子数值，其敏感度保持初始中性值。`run.json` 分别记录有效报价覆盖、活跃市场覆盖、各类缺失天数和输入标记规则。Baseline 与扩展默认使用相同训练设置，扩展增加编码器和参数；对比 IC 时仍需考虑额外模型容量，单次结果不代表稳定的增量收益。

显式缺失标记版本改变了预测头输入，固定轮数训练也升级了模型指纹，旧 Polymarket 模型需要重训。研究结束日延长后，应先补齐新区间所需原始数据、重建 A 股 provider 和 Alpha158 缓存，再运行 `uv run python -m polymarket_ext prepare --refresh` 生成覆盖新区间的因子面板，最后运行 `uv run python -m polymarket_ext train --refresh`。月度缓存指纹包含模型源码和输入版本；单独加载缺失标记升级前的模型预测时，也会因输入版本不兼容而提示重训。

区间涨跌统一后，旧分类目录、因子面板和 Polymarket 模型也必须重建；已有原始事件、价格与交易下载可复用，A 股缓存不受此次改动影响。先重跑分类，再运行普通下载补齐新纳入的多日市场，最后重建因子并训练：

```bash
uv run python -m polymarket_ext classify
uv run python -m polymarket_ext download
uv run python -m polymarket_ext prepare --refresh
uv run python -m polymarket_ext train --refresh
```

若新分类包含的所有市场行情均已下载，可跳过 `download`；现有事件快照已覆盖目标市场时无需重新 `discover`。

离线验证：`uv run pytest -q`。新增测试覆盖分类、官方分页与断点恢复、报价粒度、截面对齐、五类公式、缺口和未来信息隔离、显式缺失标记与真实零值、填充排除、敏感度梯度、未知股票与市场、模型保存及 Qlib 月度训练产物；合成数据仅用于程序测试。
