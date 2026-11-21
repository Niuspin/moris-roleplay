# 本地 llama.cpp 引擎库 (`engines/`)

本目录用于集中存放本地 `llama.cpp` 推理引擎可执行文件及依赖动态库（`.dll`）。

## 推荐目录放置方式

将编译好或从官方 Release 下载的 `llama.cpp` 解压至 `engines/llama.cpp/`（或直接放在 `engines/` 下）：

```text
engines/
└── llama.cpp/
    ├── llama-server.exe       # 推荐：HTTP 服务引擎（支持纯文本与 --mmproj 视觉多模态）
    ├── llama-cli.exe          # 可选：CLI 单轮推理引擎
    ├── llama-mtmd-cli.exe     # 可选：CLI 多模态推理引擎
    └── *.dll                  # ggml / llama / cuda / vulkan 相关运行库
```

## 快速获取方式

可双击或运行项目根目录下的 `scripts\fetch_llama_cpp.bat`，或在 Web 界面 **“2. llama.cpp 引擎库与多模态模型库”** 中点击 **“刷新库”** 自动扫描已放入的引擎可执行文件。
