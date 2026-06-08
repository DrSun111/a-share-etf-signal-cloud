# A股 ETF 短线机会评分台

本工具把 ETF 实时行情、K 线、成交额、主力资金、折溢价、行业热度、市场环境和成分股信息合并到一个 Streamlit 看板，输出「买入观察 / 持有 / 减仓 / 离场 / 禁止追高」等短线状态。

## 新增功能

- 侧边栏支持按 ETF 代码、名称、主题关键词实时查询，并从实时 ETF 列表切换当前分析标的。
- 支持基金自选池：可把当前 ETF 加入自选，也可批量编辑多个代码。
- `自选池/查询` 标签页提供自选基金实时盯盘表和全 ETF 实时查询表，展示涨跌幅、成交额、量比、主力净流入、折溢价和实时机会分。
- 自选池会保存到本目录的 `watchlist.json`；云端部署时文件持久性取决于部署平台。
- 新增云数据库模式：支持本地 SQLite 试用，也支持通过 `DATABASE_URL` 连接 Supabase/Neon/PostgreSQL。
- 新增 `collector.py`：可定时抓取实时行情和行业热度并写入数据库。

## 运行

```powershell
cd C:\Users\28050\Documents\Codex\2026-06-07\a-etf-etf-k-50-m\outputs\etf_short_signal_app
.\run_local.ps1
```

打开浏览器访问：

```text
http://localhost:8501
```

## 云端部署

GitHub 可以保存和管理代码，但 GitHub Pages 不能直接运行 Python/Streamlit。推荐把本目录上传到 GitHub 仓库，再用 Streamlit Community Cloud 部署，入口文件选择 `app.py`。详见 `DEPLOY.md`。

## 数据库试用

本地直接点击侧边栏「同步实时行情入库」，会创建 `data/etf_signal.db` 并保存最新 ETF 快照。打开「优先读取数据库快照」后，看板会优先从数据库读取实时列表。

云端部署时，在 Streamlit Cloud 或 GitHub Actions Secrets 中配置：

```text
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
```

然后用 `collector.py` 或 GitHub Actions 定时写入数据库，手机端看板读取数据库即可。

## 评分公式

```text
ETF短线强度 = 趋势分*0.25 + 量能分*0.20 + 资金分*0.20 + 板块热度分*0.20 + 风险控制分*0.15
```

- 趋势分：均线多头、突破位置、均线斜率、相对指数强弱。
- 量能分：成交额相对 5 日/20 日均额、量比、量价匹配。
- 资金分：主力净流入占比、ETF 资金排名、MFI/OBV、近 5 日吸筹代理。
- 板块热度分：行业涨幅排名、成交额排名、净流入排名、上涨家数比例。
- 风险控制分：距离前高空间、20 日回撤、折溢价、短期涨幅过热、ATR 波动。

## 数据源

- ETF 实时行情、净值、基金概况、成分股：AKShare 封装的东方财富接口。
- ETF 日 K：优先东方财富，失败时降级到新浪 ETF 历史 K 线。
- 行业热度：AKShare 封装的同花顺行业概览。
- 指数和市场宽度：优先东方财富，失败时降级到新浪指数 K 线或 ETF 宽度代理。

公网免费接口可能存在延迟、限流、字段变化和临时不可用。它适合研究和盯盘辅助；用于真实交易前，应接入授权行情源、做历史回测、加入账户级风控，并避免把评分当成单一买卖信号。
