# Moris Roleplay 扩展开发指南 (Character Skills & T2I Plugins)

Moris Roleplay 采用模块化架构，新增**角色 Skill (`.md`)** 或接入**交互后场景文生图插件 (`T2I`)** 均无需修改核心对话代码（`backend/chat_service.py`、`backend/server.py`、`app.py` 或 `start.bat`）。

---

## 一、角色 Skill (`.md`) 规范与扩展方式

角色设定由 `backend/skill_loader.py` 中的 `SkillLoader` 负责加载，支持**单个 `.md` 文件**或**包含多个 `.md` 文件的文件夹**。

### 1. 文件过滤规则（严格仅读取 `.md`）
- 当传入一个文件夹路径时，加载器**只读取扩展名为 `.md`（不区分大小写，如 `.md`、`.MD`）的 Markdown 文件**。
- 文件夹中的任何非 `.md` 文件（例如 `LICENSE`、`manifest.json`、`.png`、`.json`、`.txt`、`.bat` 等）都会被安全跳过并记录到 `ignored_files` 列表中，绝不会被读入系统提示词。
- 以 `.` 开头的隐藏目录（如 `.git/`）会被直接跳过。

### 2. YAML Frontmatter 元数据支持
可以在 `SKILL.md`（或单文件 `.md`）顶部添加可选的 YAML Frontmatter 来声明角色名称与简介：

```markdown
---
name: firefly-hsr
description: "星穹铁道角色流萤（萨姆）扮演设定"
---

# 核心角色设定
你是流萤……
```

- 加载器会自动提取 `name` 作为 `character_name`、提取 `description` 作为角色简介，并将顶部 YAML Frontmatter 从最终注入的 `system_prompt` 中剥离。
- 正文内部的 Markdown 分隔线 `---` 会完整保留。

### 3. 多文件文件夹的拼接优先级顺序
当角色 Skill 为文件夹时，加载器按以下固定顺序拼接各 `.md` 模块：
1. 核心设定文件（按序）：
   - `SKILL.md`
   - `prompt.md`
   - `profile.md`
   - `personality.md`
   - `interaction.md`
   - `relations.md`
   - `conflicts.md`
   - `memory.md`
2. 其他自定义 `.md` 文件（按文件名字母序排列）
3. 说明与评测文档（当 `exclude_meta_docs: true` 时自动排除）：
   - `README.md`
   - `quality-report.md`
   - `test-report.md`
   - `roleplay-test-report.md`
   - `CHANGELOG.md`

---

## 二、交互后场景文生图扩展接口 (`plugins/t2i`)

每一轮角色扮演对话完成后，`backend/chat_service.py` 会构造当前轮次的场景上下文 `SceneContext`，并交由 `backend/extensions/t2i.py` 中的 `T2IHookDispatcher` 分发给当前激活的文生图插件。

### 1. 核心数据结构契约

#### `SceneContext`（传入插件的场景上下文）
定义于 `backend/extensions/t2i.py`：
- `session_id: str` — 当前会话 ID
- `turn_index: int` — 当前完成的对话轮次（从 1 开始递增）
- `timestamp_utc: str` — UTC 时间戳（ISO-8601 格式）
- `character_name: str` — 当前加载的角色名称（未加载时为 `""`）
- `skill_source: str` — 当前角色 Skill 的文件或文件夹路径
- `user_message: str` — 本轮用户输入的台词/描述
- `assistant_reply: str` — 本轮模型生成的角色回复
- `recent_messages: tuple[dict[str, str], ...]` — 最近 `context_window_turns` 轮内的 `{"role": ..., "content": ...}` 消息元组
- `scene_prompt_hint: str` — 预提取的场景提示词摘要
- `metadata: dict[str, Any]` — 附加元数据（包含 `messages_count`、`skill_loaded` 等）

#### `T2IResult`（插件返回结果）
定义于 `backend/extensions/t2i.py`：
- `status: str` — 状态标识：`"disabled"` | `"skipped"` | `"stub_ok"` | `"generated"` | `"error"`
- `plugin_name: str` — 插件名称
- `invoked: bool` — 是否实际执行了插件逻辑
- `prompt_used: str` — 用于生成图片的场景提示词
- `image_url: str | None` — 生成图片的访问 URL（如接入 ComfyUI / SD WebUI 时返回，未生成时为 `None`）
- `image_path: str | None` — 生成图片的本地保存路径（未生成时为 `None`）
- `elapsed_ms: float` — 插件执行耗时（毫秒）
- `message: str` — 状态说明或错误信息
- `extra: dict[str, Any]` — 插件自定义扩展字典

### 2. 编写热插拔文生图插件 (`on_turn_completed`)

只需在 `plugins/t2i/` 目录下新建一个 `.py` 文件（文件名不以 `_` 开头），实现 `on_turn_completed(self, context: SceneContext, plugin_settings: dict[str, Any]) -> T2IResult`，并通过以下**任意一种方式**导出：

#### 方式 A：导出模块级 `PLUGIN` 实例（推荐）

```python
from backend.extensions.t2i import SceneContext, T2IResult


class MyCustomT2IPlugin:
    name = "my_custom_t2i"
    description = "自定义场景文生图插件示例"

    def on_turn_completed(self, context: SceneContext, plugin_settings: dict) -> T2IResult:
        style = plugin_settings.get("style", "anime")
        prompt = f"[{style}] {context.character_name}, {context.scene_prompt_hint}"
        # 在此处可接入本地 SD.cpp / ComfyUI / Stable Diffusion WebUI API
        return T2IResult(
            status="stub_ok",
            plugin_name=self.name,
            invoked=True,
            prompt_used=prompt,
            image_url=None,
            image_path=None,
            message="Scene prompt generated.",
            extra={"turn": context.turn_index},
        )


PLUGIN = MyCustomT2IPlugin()
```

#### 方式 B：导出 `register(registry)` 注册函数

```python
def register(registry):
    registry.register(MyCustomT2IPlugin())
```

### 3. 启用与配置插件
参考项目自带的 `plugins/t2i/example_stub.py` 与 `config.example.json`。在 `config.json`（或通过 Web 界面 / `PUT /api/config`）中更新 `t2i` 配置段即可动态切换并生效：

```json
{
  "t2i": {
    "enabled": true,
    "active_plugin": "example_stub",
    "context_window_turns": 4,
    "plugins_dir": "plugins/t2i",
    "plugin_settings": {
      "style": "anime"
    }
  }
}
```

### 4. 异常隔离保证
- 当 `t2i.enabled` 为 `false`（默认）时，`T2IHookDispatcher` 直接返回 `status="disabled", invoked=false`，不会调用任何插件。
- 当插件内部抛出异常时，`T2IHookDispatcher` 会自动捕获并返回 `status="error"` 的 `T2IResult`，绝不会影响主对话回复或导致服务崩溃。
