# Polymarket-Alternative-Data

研究 Polymarket 信息能否为 A 股收益预测提供额外因子。目前实现历史沪深300股票池上的 Qlib baseline；暂未加入 Polymarket 特征或交易回测。

## 实验口径

- 股票池：BaoStock 按指定历史日期返回的沪深300成分股，使用 **T 日成员**形成 T 日截面。来源按周更新，保存原始 `updateDate`，不将其视为精确到日的官方调整记录。
- 行情：AKShare 东方财富日线，后复权。下载研究期间所有历史成员的完整可用历史，包括研究起点前用于特征预热的行情。
- 特征与模型：Qlib Alpha158（158列）和 `LGBModel`。
- 目标：T 日收盘后预测 `close(T+2) / close(T+1) - 1`，T+1、T+2 均按全市场交易日历计算。本文以项目级设置和后续确认的一日目标为准。
- 标准化：在每个 T 日有效成员标签中使用普通 zscore，样本标准差 `ddof=1`；无效标签不进入训练，零方差截面不计分。
- 指标：逐日计算预测与标准化真实收益的 Pearson IC，按有效交易日等权汇总；同时提供 Rank IC。收益标签的 zscore 不改变 Pearson 相关系数。

不设验证集、不使用早停。每月从头训练，训练集从2008年起逐月扩展，包含测试月之前全部可用历史：

| 测试区间 | 训练历史（标签可用截止日） |
| --- | --- |
| 2026年4月 | 2008-01-01 至 2026-03-31 |
| 2026年5月 | 2008-01-01 至 2026-04-30 |
| 2026年6月 | 2008-01-01 至 2026-05-31 |
| 2026年7月 | 2008-01-01 至 2026-06-30 |
| 2026年8月 | 2008-01-01 至 2026-07-31 |
| 2026年9月1–18日 | 2008-01-01 至 2026-08-31 |

每轮仅在当前训练末尾剔除两个交易日，使所有训练标签的 T+2 不晚于测试月之前的最后一个交易日。此前月份的末尾样本在标签可用后重新纳入训练，2025年末不设永久缺口；例如7月训练包含2026年1–6月的全部已实现标签。各测试月保留月末预测，事后计算标签；最终只评价 T+2 不晚于2026-09-18的样本。最后两个交易日仍输出预测，标签为空。所有训练样本等权。

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

配置集中在 `configs/baseline.yaml`，相对路径以当前工作目录为基准。`train_start` 设置固定的训练起点，`first_test_month` 设置首个测试月，之后按月扩展训练历史。`download_end: null` 表示下载截至运行日期的可用行情；研究与评价边界仍固定为 `research_end`。`--refresh` 强制重建所选阶段产物。普通重跑复用完整且匹配的缓存；网络下载中断后可以继续运行同一命令。下载失败会记录问题并终止，避免把缺失股票静默当作有效样本池。

默认每轮1000棵迭代树，随机种子42，LightGBM使用2个线程，Qlib特征计算使用1个进程，六轮训练串行执行。历史成员查询最多使用4个独立连接，可通过 `download.membership_workers: 1` 改为串行。依赖解析锁定在 `uv.lock`。首次全量下载需要较长时间；年度特征缓存避免六轮重复计算 Alpha158。

## 数据和产物

```text
akshare/hs300/
  calendar.txt                 # 完整市场交易日，每行一个 YYYY-MM-DD
  instruments/YYYY-MM-DD.txt   # 当日成员，每行一个 SH600000/SZ000001
  data/SH600000.csv            # AKShare 后复权行情及 raw_close
  metadata.json               # 下载区间、成员来源更新日期
data/qlib/hs300/               # Qlib 日历、成员有效区间及二进制行情
data/alpha158/                 # 按年缓存特征、标准化标签及原始标签
outputs/baseline/              # 月度模型、预测、IC 和实验记录
```

行情CSV保留日期、股票代码、开高低收、成交量、成交额、振幅、涨跌幅、涨跌额、换手率，另加未复权 `raw_close`。原始成交量单位为手、成交额单位为元。转换到 Qlib 时，令 `factor=后复权收盘价/raw_close`，成交量为 `原始成交量×100/factor`，VWAP为 `成交额/(原始成交量×100)×factor`，与后复权价格保持一致。

Qlib provider 保留研究起点前的行情，但在研究结束日截断，阻止最终标签越界。停牌、零成交量和缺失行情保持缺失，不填充价格或压缩交易日历。成员退出后再次加入会保留为多个有效区间。

年度缓存采用 `(datetime, instrument)` 索引，分别保存158维特征、标准化标签及原始标签。每月装载从 `train_start` 到当轮训练截止日的完整历史，以及当月测试数据，使用 `DataHandlerLP.from_df` 和 `DatasetH` 组织数据；复用年度特征，不重复计算 Alpha158。预测不会因为未知未来标签被删除。

输出目录下 `YYYY-MM/` 保存每轮 `model.txt`、`model.pkl`、`predictions.parquet`、`daily_ic.csv`、`metrics.json`、`run.json` 和完整配置；`completed.json` 用于校验续跑产物。顶层 `predictions.parquet`、`daily_ic.csv`、`monthly_ic.csv`、`summary.json` 汇总六轮结果。预测列为 `score`、`label`（zscore标签）、`raw_label`（原始收益）。Qlib实验记录保存在本地 `mlflow.db`，对应文件位于 `artifacts/`。

## 验证与数据限制

离线测试覆盖收益对齐、截面标准化、股票池退出重入、停牌、复权量额、训练历史逐月扩展、边界标签恢复和完整 Qlib 训练链路。测试中的合成行情仅用于验证程序，不作为研究结果。

BaoStock 的历史成分股来自周度快照，可能无法反映周内临时调整的精确生效日。AKShare 公共行情源可能存在限流、退市股票历史缺失或历史数据修订；缓存元数据和失败报告用于核查实际覆盖范围，完整性检查通过后才允许训练。

接口参考：[BaoStock](https://github.com/zxygithub/baostock/blob/master/docs/pythonAPI.md)、[AKShare](https://akshare.akfamily.xyz/data/stock/stock.html)、[Qlib Alpha158](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/contrib/data/handler.py)、[LightGBM baseline配置](https://github.com/microsoft/qlib/blob/v0.9.7/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)。
