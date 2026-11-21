# 本地角色 Skill 库 (`skills/`)

本目录是项目内置的本地角色 Skill 库。同时，系统还支持在 `config.json` 的 `skill.library_dirs` 中关联外部角色 Skill 库目录（例如已默认关联 `C:\Users\nzp18\Desktop\角色skill`）。

## 角色 Skill 存放规范

你可以在 `skills/` 下按以下任一方式新增角色：
1. **角色专属文件夹**（如 `skills/firefly-skill/`）：包含 `SKILL.md`、`prompt.md`、`profile.md`、`personality.md` 等多个 `.md` 文件。加载器**严格只读取 `.md` 文件**并忽略文件夹内的任何非 `.md` 文件。
2. **单个 `.md` 文件**（如 `skills/custom_character.md`）。

放入后，在 Web 界面左侧 **“1. 角色 Skill 库 (.md)”** 点击 **“刷新库”** 即可从下拉列表中一键选择并加载。
