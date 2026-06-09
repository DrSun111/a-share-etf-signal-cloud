# 云端部署说明

## 推荐方案：GitHub + Streamlit Community Cloud

GitHub Pages 只能托管静态 HTML/CSS/JavaScript，不能直接运行 Python/Streamlit 后端。这个项目应先放到 GitHub 仓库，再用 Streamlit Community Cloud 从 GitHub 部署，部署后会得到手机可访问的 `https://...streamlit.app` 链接。

基本步骤：

1. 在 GitHub 新建仓库，例如 `a-share-etf-signal`.
2. 上传本目录全部文件：`app.py`、`requirements.txt`、`runtime.txt`、`.streamlit/config.toml` 等。
3. 打开 Streamlit Community Cloud，选择该 GitHub 仓库、分支和入口文件 `app.py`。
4. 等待部署完成，平台会生成公网 HTTPS 地址。

## 云数据库模式

推荐使用 Supabase、Neon 或 Render PostgreSQL。创建数据库后，把连接串配置为：

```text
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
```

同一条 `DATABASE_URL` 要配置到两个地方：

1. Streamlit Community Cloud -> App -> Settings -> Secrets，用于手机端看板读取云数据库。
2. GitHub 仓库 -> Settings -> Secrets and variables -> Actions -> Secrets，用于 GitHub Actions 定时写入云数据库。

App 会自动使用云数据库；本地没有 `DATABASE_URL` 时，会回退到 `data/etf_signal.db` SQLite 文件。注意：Streamlit Cloud 上的本地 SQLite 不适合长期保存，实例重启后可能丢失；长期手机端使用必须接云数据库。

本地已有 SQLite 快照时，可一次性迁移到云数据库：

```powershell
$env:DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/DBNAME"
python db_admin.py init
python db_admin.py migrate-sqlite
python db_admin.py summary
```

如需先把自选池写入云数据库：

```powershell
python db_admin.py seed-watchlist --otc "025857 110022 018345"
```

## 定时采集

本目录包含 `.github/workflows/collect.yml`。把项目推到 GitHub 后，在仓库 `Settings -> Secrets and variables -> Actions` 添加：

```text
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
OTC_WATCHLIST_CODES=025857,110022,018345
ETF_WATCHLIST_CODES=510300,159915
```

其中 `DATABASE_URL` 建议放到 Secrets；`OTC_WATCHLIST_CODES` 和 `ETF_WATCHLIST_CODES` 可放到 Variables 或 Secrets。GitHub Actions 会在 A 股交易时段每 30 分钟运行一次，也可以在 Actions 页面手动点 `Run workflow`，临时输入一组场外基金代码。

工作流会依次执行：

```powershell
python db_admin.py init
python db_admin.py seed-watchlist
python collector.py
python otc_collector.py
python db_admin.py summary
```

它会抓取 ETF 实时行情、行业热度和场外自选基金快照，并写入数据库。手机端看板开启「优先读取数据库快照」「场外自选优先读后台快照」后，会优先读取最近一次入库快照，页面刷新会快很多。

官方文档：

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/concepts/dependencies

## 临时手机测试：公网隧道

如果只想马上用手机打开本机正在运行的版本，可以使用 Cloudflare Tunnel 或 ngrok 把 `localhost:8501` 暂时映射成公网链接。临时链接适合演示，不适合作为长期生产地址。
