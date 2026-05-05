# Top100 Momentum System

一个专注“每日人气 Top100 里找持续强势股”的轻量研究系统。

## 目标

第一版只做一件事：每天把人气 Top100 当作样本池，记录当时能看到的量价和人气特征，打出“情绪持续分”，再跟踪后续 1/3/5/10 日表现。

这套系统不追求一开始就自动交易，先把研究闭环跑顺：

1. 收集每日 Top100。
2. 更新这些股票的行情缓存。
3. 生成当天特征。
4. 输出高风险观察推送。
5. 结算历史样本后续涨跌。
6. 复盘哪些特征更容易长成“圣阳股份式”样本。

## 当前数据来源

当前项目已经保存了既有 Top100 样本和对应日 K 缓存。后续增量数据默认由本项目直接抓取人气榜和相关个股日 K，本地缓存只作为加速和断网重算使用。

常用抓取命令：

```bash
python daily_job.py
python daily_job.py --capture-type intraday_0935 --snapshot-time "2026-04-20 09:35:00"
python daily_job.py --no-fetch
```

盘中快照和分钟走势默认不批量抓取；在看板“快策略”里选择感兴趣的股票后按需获取，并缓存到 `data/raw/intraday_snapshots.csv` 和 `data/raw/intraday_bars/`。实时快照接口较慢，默认只拉分钟走势，必要时再勾选“同时刷新实时快照”。

## 交易日历

主流程会先检查本地 A 股交易日历：

- 周末默认不抓新榜，只重算当前缓存。
- `data/calendar/a_share_holidays.csv` 里的节假日休市日也不抓新榜，只重算当前缓存。
- 交易日未到采集时间也会跳过抓取，例如“收盘后”要等 16:00 后才抓。

当前内置的是 2026 年沪深交易所节假日休市安排。每年交易所发布下一年度休市安排后，把新年份休市日期追加到 `data/calendar/a_share_holidays.csv` 即可；如果某年没有更新，系统仍会按周末兜底，但不能准确识别当年法定休市日。

## 常用命令

```bash
python daily_job.py
streamlit run app.py
python -m streamlit run app.py
```

`python daily_job.py` 会完成抓取、特征、信号、后验和报表输出。

原始 Top100 快照会保留盘中和收盘后的全部采集记录；日常主线样本只按每个交易日选择一个主快照，优先使用收盘后快照，没有收盘后时才使用当天最新快照。

## 日常主流程

每天只需要跑一遍主流程。可以在终端执行 `python daily_job.py`，也可以打开看板后在侧边栏点击“运行日常主流程”。

主流程固定分 5 步：

1. 抓取 Top100  
   从当前项目接口抓取每日人气 Top100，并按需更新相关个股日 K 缓存。

2. 生成特征  
   每个“日期 + 股票”生成一行研究样本，计算人气排名、首次/连续上榜、当日涨跌、近3/5日涨跌、收盘位置、量比、均线偏离、是否涨停、是否一字。

3. 情绪打分  
   根据当天能看到的信息输出 `emotion_score`、`push_level`、`reasons`、`risks`。这里是“观察推送”，不是自动买入。

4. 后验跟踪  
   对历史样本结算 1/3/5/10 日收益、最大上涨、最大回撤，用来判断规则是不是有效。

5. 更新报表  
   刷新 `fast_strategy.csv`、`fast_strategy_audit.csv`、`latest_push.csv`、`strong_recap.csv`、`rule_evaluation.csv`，看板直接读取这些简表。

快策略候选池按 `strategy_date + training_date + capture_type + snapshot_time` 锁定：同一个策略快照首次生成后会进入 `data/processed/fast_strategy_history.csv`，之后日常主流程只更新后验、审查和复盘结果，不会因为行情缓存补全而改写当时的候选池。

输出重点看这三个文件：

```text
data/reports/latest_push.csv      今日推送
data/reports/fast_strategy.csv    近一周快策略
data/reports/fast_strategy_audit.csv 快策略历史审查
data/reports/strong_recap.csv     后来涨得好的样本
data/reports/rule_evaluation.csv  规则分层表现
data/processed/fast_strategy_history.csv 快策略历史账本
```

## 目录

```text
config/settings.json        参数配置
data/raw                    原始数据副本
data/calendar               本地交易日历
data/processed              特征、信号、后验账本
data/reports                看板用简表
src                         核心逻辑
app.py                      简版看板
daily_job.py                日常主流程
```
