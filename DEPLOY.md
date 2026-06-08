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

在 Streamlit Community Cloud 的 App Secrets 中加入同名配置。App 会自动使用云数据库；本地没有 `DATABASE_URL` 时，会回退到 `data/etf_signal.db` SQLite 文件。

## 定时采集

本目录包含 `.github/workflows/collect.yml`。把项目推到 GitHub 后，在仓库 `Settings -> Secrets and variables -> Actions` 添加 `DATABASE_URL`，GitHub Actions 就可以每 30 分钟运行一次：

```powershell
python collector.py
```

它会抓取 ETF 实时行情和行业热度，并写入数据库。手机端看板开启「优先读取数据库快照」后，会优先读取最近一次入库快照。

官方文档：

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/concepts/dependencies

## 临时手机测试：公网隧道

如果只想马上用手机打开本机正在运行的版本，可以使用 Cloudflare Tunnel 或 ngrok 把 `localhost:8501` 暂时映射成公网链接。临时链接适合演示，不适合作为长期生产地址。
