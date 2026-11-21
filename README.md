# Moris Roleplay — 本地轻量化 AI 角色扮演聊天 Web 应用

**Moris Roleplay** 是一个绝对轻量化（零第三方 Python 依赖、零前端构建工具、零外部 CDN）、便于后续扩展更新的本地 AI 角色扮演聊天 Web 应用。

- **独立桌面窗口 & 无黑框启动**：双击 `MorisRoleplay.vbs` 或 `start.bat` 即可无 CMD 黑框启动独立桌面应用窗口（基于 `PySide6` / `pywebview`，不依赖本地浏览器）；关闭窗口自动清理本地模型引擎进程。
- **专属角色 Skill 库 (`skills/` + 可配置外部库)**：自动扫描项目 `skills/` 目录（及桌面可选的 `角色skill` 目录与自定义外部库），支持卡片一键切换角色；加载文件夹时**严格只读取 `.md` 文件**并自动忽略非 `.md` 文件（如 `LICENSE`、`manifest.json`、图片、`.git/` 等）。
- **专属本地 `llama.cpp` 引擎库 (`engines/`) 与多模态 GGUF 模型库 (`models/`)**：
  - 自动扫描 `engines/` 下的 `llama-server.exe` / `llama-cli.exe` / `llama-mtmd-cli.exe`；
  - 自动扫描 `models/` 下的 `.gguf` 主模型与 `*mmproj*.gguf` 视觉投影器权重，并在选择多模态模型时**自动配对绑定 `mmproj`**，启动时自动附加 `--mmproj` 参数；
  - 聊天界面支持**上传图片或 `Ctrl+V` 直接粘贴图片**发送给多模态角色模型进行视觉识别互动。
- **交互后场景文生图扩展接口 (`T2I Hook`)**：在每一轮对话完成后触发预留的场景文生图扩展钩子（默认 `enabled: false` 且 `active_plugin: "noop"`，不调用任何实际文生图模型），支持通过 `plugins/t2i/` 目录热插拔扩展自定义文生图插件。

---

## 目录结构

```text
moris roleplay/
├── start.bat                  # Windows 一键启动脚本 (检测 Python 3.11+ 并启动 app.py)
├── app.py                     # CLI 入口 (--host, --port, --config, --no-browser)
├── config.example.json        # 默认配置模板 (首次启动自动生成 config.json)
├── README.md                  # 快速开始与使用说明
├── engines/                   # 【专属引擎库】存放本地 llama.cpp 可执行文件与 DLL
│   └── llama.cpp/             # 推荐将 llama-server.exe / llama-cli.exe 放入此目录
├── models/                    # 【专属模型库】存放本地 .gguf 主模型与多模态 mmproj 权重
│   ├── llm/                   # 纯文本对话/角色扮演 .gguf 模型
│   └── multimodal/            # 多模态视觉主模型 .gguf + *mmproj*.gguf 视觉投影器
├── skills/                    # 【专属角色 Skill 库】存放项目内角色 .md 文件或文件夹
├── scripts/
│   └── fetch_llama_cpp.bat    # 可选：一键从官方 Release 下载 Windows 版 llama.cpp 到 engines/
├── docs/
│   └── EXTENSIONS.md          # 角色 Skill (.md) 与场景文生图 (T2I) 扩展开发指南
├── backend/
│   ├── config_store.py        # 原子化 JSON 配置存储与校验
│   ├── skill_loader.py        # .md 角色 Skill 加载器 (Frontmatter 解析与优先级排序)
│   ├── library_manager.py     # 本地引擎库 (engines/)、多模态模型库 (models/) 与 Skill 库扫描管理
│   ├── llama_engine.py        # llama.cpp 引擎适配器 (支持 --mmproj 多模态与图片消息块)
│   ├── chat_service.py        # 多轮会话管理、角色系统提示词注入、多模态图片转发与 T2I Hook 调度
│   ├── server.py              # ThreadingHTTPServer REST API 与本地静态资源服务
│   └── extensions/
│       └── t2i.py             # SceneContext, T2IResult, T2IPlugin, T2IRegistry, T2IHookDispatcher
├── plugins/
│   └── t2i/
│       └── example_stub.py    # 场景文生图参考演练桩插件 (仅合成提示词，不调用外部模型)
└── static/
    ├── index.html             # 本地离线单页 Web 聊天界面
    ├── style.css              # 本地离线样式表
    └── app.js                 # 原生 ES6 前端交互控制器
```

---

## 快速开始 (Quickstart)

### 1. 准备本地引擎与模型库
1. **`llama.cpp` 引擎库 (`engines/`)**：
   - 将你的 `llama.cpp` 可执行文件（`llama-server.exe` 及相关 `.dll`）放入 `engines/llama.cpp/` 目录；
   - 或直接运行 `scripts\fetch_llama_cpp.bat` 自动下载官方最新 Windows 预编译包。
2. **GGUF 与多模态模型库 (`models/`)**：
   - 纯文本模型放入 `models/llm/`（如 `Qwen2.5-7B-Instruct-Q4_K_M.gguf`）；
   - 多模态视觉模型将主模型 `.gguf` 与视觉投影器 `*mmproj*.gguf` 一并放入 `models/multimodal/`。
3. **角色 Skill 库 (`skills/`)**：
   - 将角色 `.md` 文件或文件夹放入 `skills/` 目录即可自动发现（也可在设置中心自定义任意外部目录，或使用桌面 `角色skill` 目录）。

### 2. 启动应用
双击运行 **`MorisRoleplay.vbs`**（完全无黑框）或 **`start.bat`**，或在终端中执行：

```bat
start.bat
```

启动后程序会自动在 `http://127.0.0.1:7860` 启动本地 Web 服务并打开浏览器。也可以通过 `app.py` 传入命令行参数：

```bash
python app.py --host 127.0.0.1 --port 7860 --config config.json --no-browser
```

支持的 CLI 参数（可通过 `python app.py --help` 查看）：
- `--host`：Web 服务监听地址（默认读取 `config.json` 中的 `127.0.0.1`）
- `--port`：Web 服务监听端口 `1..65535`（默认读取 `config.json` 中的 `7860`）
- `--config`：配置文件路径（默认 `config.json`，不存在时自动创建）
- `--no-browser`：启动时不自动弹出浏览器窗口
