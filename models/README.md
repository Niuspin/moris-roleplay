# 本地 GGUF 与多模态模型库 (`models/`)

本目录用于集中存放本地 `.gguf` 语言模型与多模态视觉投影器（`mmproj`）权重文件。应用启动或点击“刷新库”时会自动递归扫描本目录。

## 子目录划分建议

- `models/llm/`：存放纯文本对话/角色扮演 `.gguf` 模型（例如 `Qwen2.5-7B-Instruct-Q4_K_M.gguf`）。
- `models/multimodal/`：存放支持视觉理解的多模态主模型 `.gguf` 及其配套的视觉投影器 `*mmproj*.gguf`（或 `*clip*.gguf`）。

## 多模态 (`mmproj`) 自动配对规则

当你在 `models/multimodal/`（或同一个子文件夹）中同时放入：
1. 主模型：`Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
2. 视觉投影器：`mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf`（文件名包含 `mmproj`、`clip` 或 `vision-proj`）

模型库扫描器会自动识别并区分**主模型**与**视觉投影器 (`mmproj`)**，并在 Web 界面选择该多模态主模型时**自动关联绑定对应的 `mmproj` 文件**。启动 `llama-server` 时将自动附加 `--mmproj <path>` 参数，开启聊天界面中的图片上传与粘贴识别功能。
