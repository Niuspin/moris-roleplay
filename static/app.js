/* Moris Roleplay — Modern Vanilla ES6 Frontend Controller
 * Features: SSE streaming, abort/regenerate, roleplay rich formatting,
 *           message edit/rewind, persistent sessions, quick-ready engine,
 *           multimodal vision, and full custom theme engine.
 */
(() => {
  "use strict";

  const THEME_STORAGE_KEY = "moris_rp_theme_v3";
  const PRESET_DEFAULTS = {
    cafe: { accent: "#d4955a", bg: "#15110e", surface: "#221c17" },
    cyanotype: { accent: "#2c648a", bg: "#eff4f8", surface: "#fafdff" },
    parchment: { accent: "#9a5b32", bg: "#f5efe4", surface: "#fbf7f0" },
    sakura: { accent: "#e57ba8", bg: "#17121a", surface: "#231b28" },
    emerald: { accent: "#2cb67d", bg: "#0c1513", surface: "#142420" },
  };

  const DEFAULT_THEME_PREFS = {
    preset: "cafe",
    customColors: false,
    accent: "#d4955a",
    bg: "#15110e",
    surface: "#221c17",
    fontSize: 15,
    chatWidth: "820px",
    layout: "bubble",
    font: "sans",
  };

  const state = {
    sessionId: "session-" + Math.random().toString(36).slice(2, 10),
    turnIndex: 0,
    messages: [], // [{role, content, images?, t2iHook?}]
    sessions: [],
    characterName: "",
    userName: "旅者",
    skillPath: "",
    engineState: "stopped",
    immersionMode: false,
    busy: false,
    abortController: null,
    pendingImages: [],
    config: null,
    library: {
      engines: { items: [] },
      models: { models: [], mmprojs: [] },
      skills: { items: [] },
    },
    theme: { ...DEFAULT_THEME_PREFS },
  };

  const el = (id) => document.getElementById(id);
  let toastTimer = null;

  function showNotification(message, kind = "info") {
    const banner = el("notification-banner");
    if (!banner) return;
    banner.textContent = message;
    banner.className = `toast ${kind}`;
    banner.classList.remove("hidden");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => banner.classList.add("hidden"), 4500);
  }

  async function apiRequest(method, path, body = undefined) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    let data = null;
    try {
      data = await resp.json();
    } catch (_err) {
      data = null;
    }
    if (!resp.ok) {
      const errMsg =
        (data && data.error && (data.error.message || data.error.code)) ||
        `HTTP ${resp.status}`;
      const errDetail = data && data.error && data.error.detail ? ` (${data.error.detail})` : "";
      throw new Error(errMsg + errDetail);
    }
    return data;
  }

  function avatarInitial(name, fallback = "M") {
    const clean = (name || "").trim();
    return clean ? clean.charAt(0).toUpperCase() : fallback;
  }

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /**
   * Lightweight, XSS-safe Roleplay Markdown & Narration Formatter.
   * Highlights *action/narration*, **bold**, `code`, blockquotes, and dialogue quotes.
   */
  function formatRoleplayText(rawText) {
    const text = String(rawText || "").replace(/\r\n/g, "\n");
    if (!text.trim()) return "";

    // Extract fenced code blocks first
    const codeBlocks = [];
    const withPlaceholders = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_m, _lang, code) => {
      const idx = codeBlocks.length;
      codeBlocks.push(`<pre><code>${escapeHtml(code.trimEnd())}</code></pre>`);
      return `\u0000CODEBLOCK_${idx}\u0000`;
    });

    const paragraphs = withPlaceholders.split(/\n{2,}/);
    const htmlParts = paragraphs.map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      if (/^\u0000CODEBLOCK_(\d+)\u0000$/.test(trimmed)) {
        const idx = parseInt(trimmed.match(/\d+/)[0], 10);
        return codeBlocks[idx];
      }

      const lines = trimmed.split("\n");
      const isQuoteBlock = lines.every((ln) => ln.trim().startsWith(">"));
      if (isQuoteBlock) {
        const inner = lines.map((ln) => formatInline(ln.replace(/^\s*>\s?/, ""))).join("<br>");
        return `<blockquote>${inner}</blockquote>`;
      }

      const formattedLines = lines.map((ln) => formatInline(ln)).join("<br>");
      return `<p>${formattedLines}</p>`;
    });

    return htmlParts.join("").replace(/\u0000CODEBLOCK_(\d+)\u0000/g, (_m, i) => codeBlocks[Number(i)] || "");
  }

  function formatInline(line) {
    let s = escapeHtml(line);
    // Inline code
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Bold **text**
    s = s.replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>");
    // Roleplay action / narration *text* or （动作）
    s = s.replace(/(?<!\*)\*([^*\n]+?)\*(?!\*)/g, '<span class="rp-action">*$1*</span>');
    // Dialogue quotes 「...」 or “...”
    s = s.replace(/([「“][^」”\n]+[」”])/g, '<span class="rp-quote">$1</span>');
    return s;
  }

  // =========================================================================
  // Theme Engine
  // =========================================================================
  function loadThemePrefs() {
    try {
      const raw = localStorage.getItem(THEME_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {
          state.theme = { ...DEFAULT_THEME_PREFS, ...parsed };
        }
      }
    } catch (_e) {}
    applyTheme(state.theme, false);
  }

  function saveThemePrefs() {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(state.theme));
    } catch (_e) {}
  }

  function applyTheme(prefs, persist = true) {
    const root = document.documentElement;
    root.setAttribute("data-theme", prefs.preset || "obsidian");
    root.setAttribute("data-layout", prefs.layout || "bubble");
    root.setAttribute("data-font", prefs.font || "sans");

    root.style.setProperty("--chat-font-size", `${prefs.fontSize || 15}px`);
    root.style.setProperty("--chat-max-width", prefs.chatWidth || "820px");

    if (prefs.customColors) {
      root.style.setProperty("--accent", prefs.accent);
      root.style.setProperty("--accent-hover", prefs.accent);
      root.style.setProperty("--accent-soft", hexToRgba(prefs.accent, 0.16));
      root.style.setProperty("--bg", prefs.bg);
      root.style.setProperty("--surface", prefs.surface);
      root.style.setProperty("--assistant-bubble", prefs.surface);
    } else {
      ["--accent", "--accent-hover", "--accent-soft", "--bg", "--surface", "--assistant-bubble"].forEach(
        (prop) => root.style.removeProperty(prop)
      );
    }

    syncThemeControls();
    if (persist) saveThemePrefs();
  }

  function hexToRgba(hex, alpha) {
    const clean = (hex || "#7c6cf6").replace("#", "");
    if (clean.length !== 6) return `rgba(124,108,246,${alpha})`;
    const r = parseInt(clean.slice(0, 2), 16);
    const g = parseInt(clean.slice(2, 4), 16);
    const b = parseInt(clean.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function syncThemeControls() {
    const t = state.theme;
    document.querySelectorAll(".theme-preset-card").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.preset === t.preset && !t.customColors);
    });
    const presetDefaults = PRESET_DEFAULTS[t.preset] || PRESET_DEFAULTS.cafe;
    el("theme-accent-color").value = t.customColors ? t.accent : presetDefaults.accent;
    el("theme-bg-color").value = t.customColors ? t.bg : presetDefaults.bg;
    el("theme-surface-color").value = t.customColors ? t.surface : presetDefaults.surface;
    el("theme-font-size").value = String(t.fontSize || 15);
    el("theme-font-size-val").textContent = String(t.fontSize || 15);
    el("theme-chat-width").value = t.chatWidth || "820px";
    el("theme-layout-style").value = t.layout || "bubble";
    el("theme-font-family").value = t.font || "sans";
  }

  // =========================================================================
  // Modals & Tabs
  // =========================================================================
  function openModal(id, tabId = null) {
    const modal = el(id);
    if (!modal) return;
    modal.classList.remove("hidden");
    if (tabId) switchSettingsTab(tabId);
  }

  function closeModal(id) {
    const modal = el(id);
    if (modal) modal.classList.add("hidden");
  }

  function switchSettingsTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === tabId);
    });
    document.querySelectorAll(".tab-pane").forEach((p) => {
      p.classList.toggle("active", p.id === tabId);
    });
  }

  // =========================================================================
  // Library & UI Renderers
  // =========================================================================
  function setPill(pillEl, text, variant) {
    if (!pillEl) return;
    pillEl.textContent = text;
    pillEl.className = `pill pill-${variant}`;
  }

  function updateEngineModeVisibility() {
    const mode = el("llama-engine-mode").value;
    const managed = el("managed-fields");
    const external = el("external-fields");
    if (mode === "external_server") {
      managed.classList.add("hidden");
      external.classList.remove("hidden");
    } else {
      managed.classList.remove("hidden");
      external.classList.add("hidden");
    }
    updateMultimodalBadge();
  }

  function updateMultimodalBadge() {
    const pill = el("multimodal-mode-pill");
    const addBtn = el("btn-add-image");
    const mmEnabled = el("llama-multimodal-enabled").checked;
    const mmproj = el("llama-mmproj-path").value.trim();
    const mode = el("llama-engine-mode").value;

    if (!mmEnabled) {
      setPill(pill, "多模态关闭", "muted");
      addBtn.disabled = true;
      return;
    }
    addBtn.disabled = false;
    if (mmproj) {
      setPill(pill, "视觉多模态", "info");
    } else if (mode === "external_server") {
      setPill(pill, "外部多模态", "info");
    } else {
      setPill(pill, "纯文本", "muted");
    }
  }

  function renderLibrarySelectors(libData) {
    if (!libData) return;
    state.library = libData;

    // 1. Character cards in Sidebar + dropdown in Settings
    const skills = (libData.skills && libData.skills.items) || [];
    el("skill-lib-count").textContent = String(skills.length);

    const cardList = el("character-card-list");
    const skillSelect = el("skill-library-select");
    const currentSkillPath = state.skillPath || el("skill-path-input").value.trim();

    cardList.innerHTML = "";
    skillSelect.innerHTML = '<option value="">-- 请选择角色或在下方输入路径 --</option>';

    if (skills.length === 0) {
      cardList.innerHTML = '<div class="sidebar-empty">未在 skills/ 中发现角色 (.md)</div>';
    } else {
      skills.forEach((item) => {
        const isActive =
          currentSkillPath &&
          currentSkillPath.toLowerCase() === String(item.path || "").toLowerCase();

        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `char-item${isActive ? " active" : ""}`;
        btn.title = item.description || item.path;
        btn.innerHTML = `
          <span class="char-avatar-sm">${escapeHtml(avatarInitial(item.display_name || item.character_name, "角"))}</span>
          <span class="char-item-info">
            <div class="char-item-name">${escapeHtml(item.display_name || item.character_name)}</div>
            <div class="char-item-sub">${escapeHtml(item.library_label)} · ${item.md_file_count} .md</div>
          </span>
        `;
        btn.addEventListener("click", () => {
          el("skill-path-input").value = item.path;
          handleLoadSkill(item.path);
        });
        cardList.appendChild(btn);

        const opt = document.createElement("option");
        opt.value = item.path;
        opt.textContent = `[${item.library_label}] ${item.display_name} (${item.md_file_count} .md)`;
        skillSelect.appendChild(opt);
      });
    }
    if (currentSkillPath && [...skillSelect.options].some((o) => o.value === currentSkillPath)) {
      skillSelect.value = currentSkillPath;
    }

    // 2. Engine library
    const engines = (libData.engines && libData.engines.items) || [];
    el("engine-lib-count").textContent = String(engines.length);
    const engSelect = el("engine-library-select");
    const curEng = el("llama-cpp-path").value.trim();
    engSelect.innerHTML = '<option value="">-- 从 engines/ 库选择或手动填写 --</option>';
    engines.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.path;
      opt.textContent = `${item.name} [${item.kind}] (${item.size_mb} MB)`;
      engSelect.appendChild(opt);
    });
    if (curEng && [...engSelect.options].some((o) => o.value === curEng)) {
      engSelect.value = curEng;
    } else if (!curEng && engines.length > 0) {
      engSelect.value = engines[0].path;
      el("llama-cpp-path").value = engines[0].path;
    }

    // 3. Models library
    const models = (libData.models && libData.models.models) || [];
    el("model-lib-count").textContent = String(models.length);
    const modelSelect = el("model-library-select");
    const curModel = el("llama-model-path").value.trim();
    modelSelect.innerHTML = '<option value="">-- 从 models/ 库选择主模型或手动填写 --</option>';
    models.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.path;
      opt.dataset.suggestedMmproj = item.suggested_mmproj || "";
      const tag = item.modality === "multimodal" ? "多模态" : "纯文本";
      opt.textContent = `[${tag}] ${item.rel_path} (${item.size_mb} MB)`;
      modelSelect.appendChild(opt);
    });
    if (curModel && [...modelSelect.options].some((o) => o.value === curModel)) {
      modelSelect.value = curModel;
    }

    // 4. mmproj library
    const mmprojs = (libData.models && libData.models.mmprojs) || [];
    el("mmproj-lib-count").textContent = String(mmprojs.length);
    const mmSelect = el("mmproj-library-select");
    const curMm = el("llama-mmproj-path").value.trim();
    mmSelect.innerHTML = '<option value="">-- 纯文本模式 (不挂载 mmproj) --</option>';
    mmprojs.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.path;
      opt.textContent = `[mmproj] ${item.rel_path} (${item.size_mb} MB)`;
      mmSelect.appendChild(opt);
    });
    if (curMm && [...mmSelect.options].some((o) => o.value === curMm)) {
      mmSelect.value = curMm;
    }

    updateEngineMiniModelText();
    updateMultimodalBadge();
  }

  function updateEngineMiniModelText() {
    const modelPath = el("llama-model-path").value.trim();
    const mode = el("llama-engine-mode").value;
    const sub = el("engine-mini-model");
    if (!sub) return;
    if (mode === "external_server") {
      sub.textContent = el("llama-server-url").value.trim() || "外部服务模式";
    } else if (modelPath) {
      const parts = modelPath.split(/[\\/]/);
      sub.textContent = parts[parts.length - 1];
    } else {
      sub.textContent = "未选择模型 · 点击设置配置";
    }
  }

  function renderSkillSummary(skillData) {
    const box = el("skill-summary-box");
    const badge = el("skill-status-badge");
    const titleEl = el("active-character-title");
    const descEl = el("active-character-desc");
    const avatarEl = el("active-char-avatar");
    const welcomeAvatar = el("welcome-avatar");
    const welcomeTitle = el("welcome-title");

    if (!skillData || skillData.loaded === false || !skillData.character_name) {
      box.classList.add("hidden");
      setPill(badge, "未加载", "muted");
      titleEl.textContent = "未加载角色 Skill";
      descEl.textContent = "在左侧点击角色卡片即可一键进入角色扮演";
      avatarEl.textContent = "M";
      if (welcomeAvatar) welcomeAvatar.textContent = "M";
      if (welcomeTitle) welcomeTitle.textContent = "欢迎来到 Moris Roleplay";
      state.characterName = "";
      state.skillPath = "";
      return;
    }

    state.characterName = skillData.character_name;
    state.skillPath = skillData.source_path || "";
    const initial = avatarInitial(skillData.character_name, "角");

    setPill(badge, "已加载", "success");
    box.classList.remove("hidden");
    avatarEl.textContent = initial;
    if (welcomeAvatar) welcomeAvatar.textContent = initial;
    if (welcomeTitle) welcomeTitle.textContent = `与 ${skillData.character_name} 开始对话`;

    el("skill-char-name").textContent = skillData.character_name;
    el("skill-source-type").textContent =
      skillData.source_type === "directory" ? "文件夹 (.md)" : "单文件 (.md)";
    el("skill-stats").textContent = `${skillData.total_chars || 0} chars / ~${
      skillData.estimated_tokens || 0
    } tokens`;

    const loadedFiles = skillData.loaded_files || [];
    el("skill-loaded-count").textContent = String(loadedFiles.length);
    const loadedWrap = el("skill-loaded-files");
    loadedWrap.innerHTML = "";
    loadedFiles.forEach((f) => {
      const span = document.createElement("span");
      span.className = "file-pill";
      span.textContent = f;
      loadedWrap.appendChild(span);
    });

    const ignoredFiles = skillData.ignored_files || [];
    el("skill-ignored-count").textContent = String(ignoredFiles.length);
    const ignoredWrap = el("skill-ignored-files");
    ignoredWrap.innerHTML = "";
    ignoredFiles.forEach((f) => {
      const span = document.createElement("span");
      span.className = "file-pill";
      span.textContent = f;
      ignoredWrap.appendChild(span);
    });

    const warnings = skillData.warnings || [];
    const warnBox = el("skill-warnings");
    if (warnings.length > 0) {
      warnBox.textContent = warnings.join(" | ");
      warnBox.classList.remove("hidden");
    } else {
      warnBox.classList.add("hidden");
    }

    el("skill-prompt-preview").textContent = skillData.system_prompt || "";
    titleEl.textContent = skillData.character_name;
    descEl.textContent =
      skillData.description ||
      `已加载 ${loadedFiles.length} 个 .md 设定文件 (~${skillData.estimated_tokens || 0} tokens)`;

    // Highlight active character card
    renderLibrarySelectors(state.library);
  }

  function renderEngineStatus(eng) {
    if (!eng) return;
    const st = eng.state || "stopped";
    state.engineState = st;

    const dot = el("engine-status-dot");
    const label = el("engine-status-badge");
    const quickBtn = el("btn-quick-toggle-engine");

    dot.className = `status-dot dot-${st}`;
    const labelMap = {
      ready: "引擎就绪",
      starting: "启动中...",
      error: "引擎异常",
      stopped: "未启动",
    };
    label.textContent = labelMap[st] || st;

    if (st === "ready") {
      quickBtn.textContent = "停止";
      quickBtn.className = "btn btn-xs btn-danger";
    } else if (st === "starting") {
      quickBtn.textContent = "启动中";
      quickBtn.className = "btn btn-xs btn-secondary";
    } else {
      quickBtn.textContent = "一键就绪";
      quickBtn.className = "btn btn-xs btn-primary";
    }

    const parts = [`状态: ${st}`];
    if (eng.mode) parts.push(`模式: ${eng.mode}`);
    if (eng.endpoint) parts.push(`端点: ${eng.endpoint}`);
    if (eng.pid) parts.push(`PID: ${eng.pid}`);
    if (eng.error_code) parts.push(`错误码: ${eng.error_code}`);
    if (eng.detail) parts.push(eng.detail);
    el("engine-detail-line").textContent = parts.join(" · ");
  }

  function renderT2IStatus(t2iSummary) {
    if (!t2iSummary) return;
    const badge = el("t2i-status-badge");
    if (t2iSummary.enabled) {
      setPill(badge, `已启用 (${t2iSummary.active_plugin})`, "success");
    } else {
      setPill(badge, "未启用", "muted");
    }
    const select = el("t2i-active-plugin");
    const curVal = t2iSummary.active_plugin || select.value || "noop";
    const plugins = t2iSummary.plugins || [];
    if (plugins.length > 0) {
      select.innerHTML = "";
      plugins.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.description ? `${p.name} — ${p.description}` : p.name;
        select.appendChild(opt);
      });
      if (![...select.options].some((o) => o.value === curVal)) {
        const extra = document.createElement("option");
        extra.value = curVal;
        extra.textContent = curVal;
        select.appendChild(extra);
      }
      select.value = curVal;
    }
  }

  function renderSessionList(sessions) {
    state.sessions = sessions || [];
    el("session-list-count").textContent = String(state.sessions.length);
    const listEl = el("session-history-list");
    listEl.innerHTML = "";

    if (state.sessions.length === 0) {
      listEl.innerHTML = '<div class="sidebar-empty">暂无历史对话</div>';
      return;
    }

    state.sessions.forEach((s) => {
      const item = document.createElement("div");
      item.className = `session-item${s.session_id === state.sessionId ? " active" : ""}`;
      item.innerHTML = `
        <div class="session-item-body">
          <div class="session-item-title">${escapeHtml(s.title || "新对话")}</div>
          <div class="session-item-meta">${escapeHtml(s.character_name || "通用")} · ${s.turn_index || 0} 轮</div>
        </div>
        <button type="button" class="session-del-btn" title="删除会话">
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      `;
      item.addEventListener("click", (ev) => {
        if (ev.target.closest(".session-del-btn")) return;
        loadSessionHistory(s.session_id);
      });
      item.querySelector(".session-del-btn").addEventListener("click", (ev) => {
        ev.stopPropagation();
        handleDeleteSession(s.session_id);
      });
      listEl.appendChild(item);
    });
  }

  function populateConfigForm(cfg) {
    if (!cfg) return;
    state.config = cfg;
    if (cfg.skill) {
      if (cfg.skill.last_path && !el("skill-path-input").value) {
        el("skill-path-input").value = cfg.skill.last_path;
      }
      el("skill-exclude-meta").checked = Boolean(cfg.skill.exclude_meta_docs);
      el("skill-max-chars").value = String(cfg.skill.max_prompt_chars ?? 0);

      state.userName = cfg.skill.user_name || "旅者";
      state.immersionMode = Boolean(cfg.skill.immersion_mode);
      el("rp-user-name").value = state.userName;
      el("rp-user-persona").value = cfg.skill.user_persona || "";
      el("rp-scenario").value = cfg.skill.scenario || "";
      el("rp-authors-note").value = cfg.skill.authors_note || "";
      el("rp-authors-note-depth").value = String(cfg.skill.authors_note_depth ?? 2);
      el("rp-immersion-mode").checked = state.immersionMode;

      // Sync quick director drawer & header immersion pill
      el("quick-scenario").value = cfg.skill.scenario || "";
      el("quick-authors-note").value = cfg.skill.authors_note || "";
      el("btn-toggle-immersion").classList.toggle("active", state.immersionMode);
      el("cfg-skill-dirs").value = Array.isArray(cfg.skill.library_dirs)
        ? cfg.skill.library_dirs.join("\n")
        : "skills";
    }
    if (cfg.llama) {
      el("llama-engine-mode").value = cfg.llama.engine_mode || "managed_server";
      el("cfg-engine-dir").value = cfg.llama.engine_dir || "engines";
      el("cfg-models-dir").value = cfg.llama.models_dir || "models";
      el("llama-cpp-path").value = cfg.llama.llama_cpp_path || "";
      el("llama-model-path").value = cfg.llama.model_path || "";
      el("llama-mmproj-path").value = cfg.llama.mmproj_path || "";
      el("llama-multimodal-enabled").checked =
        cfg.llama.multimodal_enabled !== undefined ? Boolean(cfg.llama.multimodal_enabled) : true;
      el("llama-cache-prompt").checked =
        cfg.llama.cache_prompt !== undefined ? Boolean(cfg.llama.cache_prompt) : true;
      el("llama-server-url").value = cfg.llama.server_url || "http://127.0.0.1:8080";
      el("llama-port").value = String(cfg.llama.port ?? 8080);
      el("llama-ctx-size").value = String(cfg.llama.ctx_size ?? 16384);
      el("llama-ngl").value = String(cfg.llama.n_gpu_layers ?? -1);
      el("llama-temperature").value = String(cfg.llama.temperature ?? 0.8);
      el("llama-top-p").value = String(cfg.llama.top_p ?? 0.9);
      el("llama-min-p").value = String(cfg.llama.min_p ?? 0.05);
      el("llama-repeat-penalty").value = String(cfg.llama.repeat_penalty ?? 1.12);
      el("llama-presence-penalty").value = String(cfg.llama.presence_penalty ?? 0.1);
      el("llama-max-tokens").value = String(cfg.llama.max_tokens ?? 512);
      el("llama-max-history").value = String(cfg.llama.max_history_turns ?? 20);
      updateEngineModeVisibility();
      updateEngineMiniModelText();
    }
    if (cfg.t2i) {
      el("t2i-enabled").checked = Boolean(cfg.t2i.enabled);
      el("t2i-context-turns").value = String(cfg.t2i.context_window_turns ?? 4);
      el("t2i-plugin-settings").value = JSON.stringify(cfg.t2i.plugin_settings || {}, null, 2);
    }
  }

  function collectConfigPatch() {
    let pluginSettings = {};
    const rawSettings = el("t2i-plugin-settings").value.trim();
    if (rawSettings) pluginSettings = JSON.parse(rawSettings);

    const skillDirs = (el("cfg-skill-dirs").value || "skills")
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);

    return {
      skill: {
        last_path: el("skill-path-input").value.trim(),
        exclude_meta_docs: el("skill-exclude-meta").checked,
        max_prompt_chars: parseInt(el("skill-max-chars").value || "0", 10),
        library_dirs: skillDirs.length > 0 ? skillDirs : ["skills"],
        user_name: el("rp-user-name").value.trim() || "旅者",
        user_persona: el("rp-user-persona").value.trim(),
        scenario: el("rp-scenario").value.trim(),
        authors_note: el("rp-authors-note").value.trim(),
        authors_note_depth: parseInt(el("rp-authors-note-depth").value || "2", 10),
        immersion_mode: el("rp-immersion-mode").checked,
      },
      llama: {
        engine_mode: el("llama-engine-mode").value,
        engine_dir: el("cfg-engine-dir").value.trim() || "engines",
        models_dir: el("cfg-models-dir").value.trim() || "models",
        llama_cpp_path: el("llama-cpp-path").value.trim(),
        model_path: el("llama-model-path").value.trim(),
        mmproj_path: el("llama-mmproj-path").value.trim(),
        multimodal_enabled: el("llama-multimodal-enabled").checked,
        cache_prompt: el("llama-cache-prompt").checked,
        server_url: el("llama-server-url").value.trim(),
        port: parseInt(el("llama-port").value || "8080", 10),
        ctx_size: parseInt(el("llama-ctx-size").value || "16384", 10),
        n_gpu_layers: parseInt(el("llama-ngl").value || "-1", 10),
        temperature: parseFloat(el("llama-temperature").value || "0.8"),
        top_p: parseFloat(el("llama-top-p").value || "0.9"),
        min_p: parseFloat(el("llama-min-p").value || "0.05"),
        repeat_penalty: parseFloat(el("llama-repeat-penalty").value || "1.12"),
        presence_penalty: parseFloat(el("llama-presence-penalty").value || "0.1"),
        max_tokens: parseInt(el("llama-max-tokens").value || "512", 10),
        max_history_turns: parseInt(el("llama-max-history").value || "20", 10),
      },
      t2i: {
        enabled: el("t2i-enabled").checked,
        active_plugin: el("t2i-active-plugin").value || "noop",
        context_window_turns: parseInt(el("t2i-context-turns").value || "4", 10),
        plugin_settings: pluginSettings,
      },
    };
  }

  // =========================================================================
  // Chat Rendering & Message Operations
  // =========================================================================
  function renderChatTranscript() {
    const container = el("chat-messages");
    const placeholder = el("empty-chat-placeholder");

    // Clear existing message rows while keeping welcome placeholder
    Array.from(container.querySelectorAll(".message-row")).forEach((node) => node.remove());

    if (state.messages.length === 0) {
      if (placeholder) placeholder.classList.remove("hidden");
      el("turn-counter-label").textContent = "轮次 0";
      return;
    }

    if (placeholder) placeholder.classList.add("hidden");
    state.messages.forEach((msg, idx) => {
      const row = createMessageElement(msg, idx);
      container.appendChild(row);
    });
    el("turn-counter-label").textContent = `轮次 ${state.turnIndex}`;
    container.scrollTop = container.scrollHeight;
  }

  function createMessageElement(msg, index, isStreaming = false) {
    const isUser = msg.role === "user";
    const author = isUser ? state.userName || "你" : state.characterName || "角色";
    const row = document.createElement("div");
    row.className = `message-row ${isUser ? "role-user" : "role-assistant"}`;
    row.dataset.index = String(index);

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = avatarInitial(author, isUser ? "你" : "角");

    const col = document.createElement("div");
    col.className = "msg-column";

    const header = document.createElement("div");
    header.className = "msg-header";
    const authorEl = document.createElement("span");
    authorEl.className = "msg-author";
    authorEl.textContent = author;
    header.appendChild(authorEl);

    if (msg.speedText) {
      const speedSpan = document.createElement("span");
      speedSpan.textContent = msg.speedText;
      header.appendChild(speedSpan);
    }

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";

    const contentEl = document.createElement("div");
    contentEl.className = `msg-content${isStreaming ? " streaming-cursor" : ""}`;
    contentEl.innerHTML = formatRoleplayText(msg.content);
    bubble.appendChild(contentEl);

    if (msg.images && msg.images.length > 0) {
      const imgRow = document.createElement("div");
      imgRow.className = "message-images";
      msg.images.forEach((src) => {
        const img = document.createElement("img");
        img.src = src;
        img.alt = "多模态图片";
        img.addEventListener("click", () => openLightbox(src));
        imgRow.appendChild(img);
      });
      bubble.appendChild(imgRow);
    }

    if (msg.t2iHook && msg.t2iHook.invoked) {
      const hookCard = document.createElement("div");
      hookCard.className = "t2i-hook-card";
      hookCard.textContent = `[场景文生图 · ${msg.t2iHook.plugin_name}] ${msg.t2iHook.status}${
        msg.t2iHook.prompt_used ? " — " + msg.t2iHook.prompt_used : ""
      }`;
      bubble.appendChild(hookCard);
    }

    const actions = document.createElement("div");
    actions.className = "msg-actions";
    if (!isStreaming) {
      // Copy
      const copyBtn = makeActionBtn("复制", () => {
        navigator.clipboard.writeText(msg.content || "").then(() => showNotification("已复制到剪贴板", "success"));
      });
      actions.appendChild(copyBtn);

      // Edit
      const editBtn = makeActionBtn("编辑", () => startInlineEdit(row, index));
      actions.appendChild(editBtn);

      // Regenerate if last assistant message
      if (!isUser && index === state.messages.length - 1) {
        const regenBtn = makeActionBtn("重新生成", () => handleRegenerate());
        actions.appendChild(regenBtn);
      }

      // Rewind / delete from here
      const rewindBtn = makeActionBtn("回溯删除", () => handleTruncateFrom(index), true);
      actions.appendChild(rewindBtn);
    }

    col.appendChild(header);
    col.appendChild(bubble);
    col.appendChild(actions);
    row.appendChild(avatar);
    row.appendChild(col);
    return row;
  }

  function makeActionBtn(label, onClick, isDanger = false) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `msg-act-btn${isDanger ? " danger" : ""}`;
    btn.textContent = label;
    btn.addEventListener("click", onClick);
    return btn;
  }

  function startInlineEdit(rowEl, index) {
    if (state.busy) return;
    const msg = state.messages[index];
    if (!msg) return;
    const bubble = rowEl.querySelector(".msg-bubble");
    const origHtml = bubble.innerHTML;

    bubble.innerHTML = "";
    const box = document.createElement("div");
    box.className = "msg-edit-box";
    const ta = document.createElement("textarea");
    ta.className = "msg-edit-textarea";
    ta.value = msg.content;
    const actRow = document.createElement("div");
    actRow.className = "msg-edit-actions";

    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "btn btn-ghost btn-xs";
    cancelBtn.textContent = "取消";
    cancelBtn.addEventListener("click", () => {
      bubble.innerHTML = origHtml;
    });

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "btn btn-primary btn-xs";
    saveBtn.textContent = "保存修改";
    saveBtn.addEventListener("click", async () => {
      const updatedText = ta.value.trim();
      if (!updatedText) return;
      try {
        await apiRequest("PUT", "/api/sessions/message", {
          session_id: state.sessionId,
          index,
          content: updatedText,
        });
        state.messages[index].content = updatedText;
        renderChatTranscript();
        showNotification("消息已更新", "success");
      } catch (err) {
        showNotification(`编辑失败: ${err.message}`, "error");
      }
    });

    actRow.appendChild(cancelBtn);
    actRow.appendChild(saveBtn);
    box.appendChild(ta);
    box.appendChild(actRow);
    bubble.appendChild(box);
    ta.focus();
  }

  async function handleTruncateFrom(index) {
    if (state.busy) return;
    try {
      const res = await apiRequest("POST", "/api/sessions/truncate", {
        session_id: state.sessionId,
        index,
      });
      state.messages = (res.messages || []).map((m) => ({ role: m.role, content: m.content }));
      state.turnIndex = res.turn_index || 0;
      renderChatTranscript();
      refreshSessions();
      showNotification("已回溯到该节点", "info");
    } catch (err) {
      showNotification(`回溯失败: ${err.message}`, "error");
    }
  }

  function openLightbox(src) {
    el("lightbox-img").src = src;
    el("lightbox-modal").classList.remove("hidden");
  }

  // =========================================================================
  // Multimodal Images
  // =========================================================================
  function renderPendingImages() {
    const strip = el("chat-image-previews");
    strip.innerHTML = "";
    if (state.pendingImages.length === 0) {
      strip.classList.add("hidden");
      return;
    }
    strip.classList.remove("hidden");
    state.pendingImages.forEach((dataUrl, idx) => {
      const wrap = document.createElement("div");
      wrap.className = "preview-thumb-item";
      const img = document.createElement("img");
      img.src = dataUrl;
      img.alt = `附加图片 ${idx + 1}`;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preview-remove-btn";
      btn.textContent = "×";
      btn.addEventListener("click", () => {
        state.pendingImages.splice(idx, 1);
        renderPendingImages();
      });
      wrap.appendChild(img);
      wrap.appendChild(btn);
      strip.appendChild(wrap);
    });
  }

  function addImageFiles(fileList) {
    if (!el("llama-multimodal-enabled").checked) {
      showNotification("请先在设置中心启用视觉多模态图片输入", "error");
      return;
    }
    Array.from(fileList || []).forEach((file) => {
      if (!file.type.startsWith("image/")) return;
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          state.pendingImages.push(reader.result);
          renderPendingImages();
        }
      };
      reader.readAsDataURL(file);
    });
  }

  // =========================================================================
  // SSE Streaming Chat & Regenerate
  // =========================================================================
  function setStreamingUI(active) {
    state.busy = active;
    el("btn-send-chat").classList.toggle("hidden", active);
    el("btn-stop-stream").classList.toggle("hidden", !active);
    el("btn-regenerate-last").disabled = active;
  }

  async function consumeSseStream(url, payload, assistantMsgObj, contentEl, rowEl) {
    const controller = new AbortController();
    state.abortController = controller;
    let donePayload = null;

    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!resp.ok) {
      let errData = null;
      try {
        errData = await resp.json();
      } catch (_e) {}
      const msg =
        (errData && errData.error && (errData.error.message || errData.error.code)) ||
        `HTTP ${resp.status}`;
      throw new Error(msg);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    const container = el("chat-messages");

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";

      for (const frame of frames) {
        const line = frame
          .split("\n")
          .find((ln) => ln.startsWith("data:"));
        if (!line) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;
        let ev = null;
        try {
          ev = JSON.parse(jsonStr);
        } catch (_e) {
          continue;
        }
        if (ev.type === "delta") {
          assistantMsgObj.content += ev.content || "";
          contentEl.innerHTML = formatRoleplayText(assistantMsgObj.content);
          container.scrollTop = container.scrollHeight;
        } else if (ev.type === "error") {
          const errMsg = (ev.error && (ev.error.message || ev.error.code)) || "流式生成出错";
          const errDetail = ev.error && ev.error.detail ? ` (${ev.error.detail})` : "";
          throw new Error(errMsg + errDetail);
        } else if (ev.type === "done") {
          donePayload = ev;
        }
      }
    }
    return donePayload;
  }

  async function handleSendChat(overrideText = null) {
    if (state.busy) return;
    const inputEl = el("chat-input");
    const msg = (overrideText !== null ? overrideText : inputEl.value).trim();
    if (!msg) return;

    const imagesToSend = state.pendingImages.slice();
    state.pendingImages = [];
    renderPendingImages();
    if (overrideText === null) {
      inputEl.value = "";
      autoResizeTextarea(inputEl);
    }

    const userMsgObj = { role: "user", content: msg, images: imagesToSend };
    const assistantMsgObj = { role: "assistant", content: "" };
    state.messages.push(userMsgObj);
    state.messages.push(assistantMsgObj);

    const placeholder = el("empty-chat-placeholder");
    if (placeholder) placeholder.classList.add("hidden");

    const container = el("chat-messages");
    container.appendChild(createMessageElement(userMsgObj, state.messages.length - 2, false));
    const streamRow = createMessageElement(assistantMsgObj, state.messages.length - 1, true);
    container.appendChild(streamRow);
    container.scrollTop = container.scrollHeight;

    const contentEl = streamRow.querySelector(".msg-content");
    if (state.engineState !== "ready") {
      contentEl.innerHTML = '<span class="rp-action">正在唤醒本地模型引擎，请稍候...</span>';
    }

    setStreamingUI(true);
    if (state.engineState !== "ready") {
      try {
        const readyRes = await apiRequest("POST", "/api/quick-ready", {});
        if (readyRes.config) populateConfigForm(readyRes.config);
        if (readyRes.skill) renderSkillSummary(readyRes.skill);
        if (readyRes.engine) renderEngineStatus(readyRes.engine);
        if (readyRes.notice) showNotification(readyRes.notice, "info");
        contentEl.innerHTML = '<span class="rp-action">正在思考回复...</span>';
      } catch (readyErr) {
        state.messages.pop();
        state.messages.pop();
        renderChatTranscript();
        setStreamingUI(false);
        refreshEngineStatusQuietly();
        showNotification(`自动唤醒引擎失败: ${readyErr.message}`, "error");
        return;
      }
    }

    const payload = { session_id: state.sessionId, message: msg };
    if (imagesToSend.length > 0) payload.images = imagesToSend;

    try {
      const done = await consumeSseStream(
        "/api/chat/stream",
        payload,
        assistantMsgObj,
        contentEl,
        streamRow
      );
      if (done) {
        if (done.reply !== undefined && !assistantMsgObj.content) {
          assistantMsgObj.content = done.reply;
        }
        state.turnIndex = done.turn_index ?? state.turnIndex + 1;
        assistantMsgObj.t2iHook = done.t2i_hook || null;

        const speedStr = formatSpeedMetric(done);
        if (speedStr) {
          assistantMsgObj.speedText = speedStr;
          el("speed-metric-label").textContent = speedStr;
        }
        if (done.t2i_hook) {
          el("t2i-last-status").textContent = `${done.t2i_hook.status} (${done.t2i_hook.plugin_name})`;
          el("t2i-last-prompt").textContent = done.t2i_hook.prompt_used || done.t2i_hook.message || "";
        }
      }
      renderChatTranscript();
      refreshSessions();
      refreshEngineStatusQuietly();
    } catch (err) {
      if (err.name === "AbortError") {
        showNotification("已停止生成", "info");
        if (!assistantMsgObj.content.trim()) {
          state.messages.pop();
        }
        renderChatTranscript();
      } else {
        state.messages.pop(); // remove empty assistant placeholder
        state.messages.pop(); // revert optimistic user message
        renderChatTranscript();
        showNotification(`发送失败: ${err.message}`, "error");
        refreshEngineStatusQuietly();
      }
    } finally {
      state.abortController = null;
      setStreamingUI(false);
      inputEl.focus();
    }
  }

  async function handleRegenerate() {
    if (state.busy || state.messages.length === 0) return;
    if (state.messages[state.messages.length - 1].role === "assistant") {
      state.messages.pop();
    }
    const assistantMsgObj = { role: "assistant", content: "" };
    state.messages.push(assistantMsgObj);
    renderChatTranscript();

    const rows = el("chat-messages").querySelectorAll(".message-row");
    const lastRow = rows[rows.length - 1];
    const contentEl = lastRow.querySelector(".msg-content");
    contentEl.classList.add("streaming-cursor");

    setStreamingUI(true);
    try {
      const done = await consumeSseStream(
        "/api/chat/regenerate/stream",
        { session_id: state.sessionId },
        assistantMsgObj,
        contentEl,
        lastRow
      );
      if (done) {
        if (done.reply !== undefined && !assistantMsgObj.content) {
          assistantMsgObj.content = done.reply;
        }
        state.turnIndex = done.turn_index ?? state.turnIndex;
        assistantMsgObj.t2iHook = done.t2i_hook || null;
        const speedStr = formatSpeedMetric(done);
        if (speedStr) {
          assistantMsgObj.speedText = speedStr;
          el("speed-metric-label").textContent = speedStr;
        }
      }
      renderChatTranscript();
      refreshSessions();
    } catch (err) {
      if (err.name === "AbortError") {
        showNotification("已停止重新生成", "info");
      } else {
        showNotification(`重新生成失败: ${err.message}`, "error");
      }
      await loadSessionHistory(state.sessionId);
    } finally {
      state.abortController = null;
      setStreamingUI(false);
    }
  }

  function formatSpeedMetric(done) {
    if (!done) return "";
    const timings = done.timings || {};
    if (typeof timings.predicted_per_second === "number" && timings.predicted_per_second > 0) {
      return `${timings.predicted_per_second.toFixed(1)} tok/s`;
    }
    const compTokens = done.usage && done.usage.completion_tokens;
    const elapsedMs = done.elapsed_ms;
    if (compTokens && elapsedMs && elapsedMs > 0) {
      return `${(compTokens / (elapsedMs / 1000)).toFixed(1)} tok/s`;
    }
    if (elapsedMs && elapsedMs > 0) {
      return `${(elapsedMs / 1000).toFixed(1)}s`;
    }
    return "";
  }

  // =========================================================================
  // Session & Quick Actions
  // =========================================================================
  async function refreshSessions() {
    try {
      const res = await apiRequest("GET", "/api/sessions");
      renderSessionList(res.sessions || []);
    } catch (_e) {}
  }

  async function loadSessionHistory(sessionId) {
    if (state.busy) return;
    try {
      const detail = await apiRequest(
        "GET",
        `/api/sessions/history?session_id=${encodeURIComponent(sessionId)}`
      );
      state.sessionId = detail.session_id || sessionId;
      state.turnIndex = detail.turn_index || 0;
      state.messages = (detail.messages || []).map((m) => ({ role: m.role, content: m.content }));
      renderChatTranscript();
      renderSessionList(state.sessions);
    } catch (err) {
      showNotification(`加载会话失败: ${err.message}`, "error");
    }
  }

  function handleNewSession() {
    if (state.busy) return;
    state.sessionId = "session-" + Math.random().toString(36).slice(2, 10);
    state.turnIndex = 0;
    state.messages = [];
    state.pendingImages = [];
    renderPendingImages();
    renderChatTranscript();
    renderSessionList(state.sessions);
    el("chat-input").focus();
  }

  async function handleDeleteSession(sessionId) {
    try {
      await apiRequest("POST", "/api/sessions/delete", { session_id: sessionId });
      if (sessionId === state.sessionId) {
        handleNewSession();
      }
      await refreshSessions();
      showNotification("会话已删除", "info");
    } catch (err) {
      showNotification(`删除会话失败: ${err.message}`, "error");
    }
  }

  async function handleResetSession() {
    if (state.busy) return;
    try {
      await apiRequest("POST", "/api/sessions/reset", { session_id: state.sessionId });
      state.turnIndex = 0;
      state.messages = [];
      state.pendingImages = [];
      renderPendingImages();
      renderChatTranscript();
      refreshSessions();
      showNotification("当前对话记录已清空（已保留当前角色）", "success");
    } catch (err) {
      showNotification(`清空失败: ${err.message}`, "error");
    }
  }

  function handleExportMarkdown() {
    if (state.messages.length === 0) {
      showNotification("当前对话为空，无需导出", "info");
      return;
    }
    const charName = state.characterName || "角色";
    const userName = state.userName || "你";
    const lines = [`# ${charName} — 对话记录\n`];
    state.messages.forEach((m) => {
      const speaker = m.role === "user" ? userName : charName;
      lines.push(`### ${speaker}\n\n${m.content}\n`);
    });
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `moris-${charName}-${state.sessionId.slice(-6)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // =========================================================================
  // Backend State & Engine Controls
  // =========================================================================
  async function refreshEngineStatusQuietly() {
    try {
      const st = await apiRequest("GET", "/api/engine/status");
      renderEngineStatus(st);
    } catch (_e) {}
  }

  async function refreshAllState() {
    try {
      const [health, cfg, t2i, lib, sessRes] = await Promise.all([
        apiRequest("GET", "/api/health"),
        apiRequest("GET", "/api/config"),
        apiRequest("GET", "/api/extensions/t2i"),
        apiRequest("GET", "/api/library"),
        apiRequest("GET", "/api/sessions"),
      ]);
      setPill(el("health-badge"), "在线", "success");
      populateConfigForm(cfg);
      renderLibrarySelectors(lib);
      renderSkillSummary(health.skill);
      renderEngineStatus(health.engine);
      renderT2IStatus(t2i);
      renderSessionList(sessRes.sessions || []);
    } catch (err) {
      setPill(el("health-badge"), "离线", "danger");
      showNotification(`无法连接后端服务: ${err.message}`, "error");
    }
  }

  async function handleQuickReady() {
    const btns = [el("btn-quick-toggle-engine"), el("btn-welcome-quick-ready")].filter(Boolean);
    if (state.engineState === "ready") {
      await handleStopEngine();
      return;
    }
    btns.forEach((b) => (b.disabled = true));
    renderEngineStatus({ state: "starting", mode: el("llama-engine-mode").value });
    showNotification("正在自动加载角色并启动本地模型引擎，请稍候...", "info");
    try {
      const res = await apiRequest("POST", "/api/quick-ready", {});
      if (res.config) populateConfigForm(res.config);
      if (res.skill) renderSkillSummary(res.skill);
      if (res.engine) renderEngineStatus(res.engine);
      showNotification(res.notice || "已就绪！现在可以直接开始角色扮演对话", "success");
    } catch (err) {
      refreshEngineStatusQuietly();
      showNotification(`一键就绪失败: ${err.message}`, "error");
    } finally {
      btns.forEach((b) => (b.disabled = false));
    }
  }

  async function handleRescanLibrary() {
    el("btn-rescan-library").disabled = true;
    try {
      const lib = await apiRequest("POST", "/api/library/rescan", {});
      renderLibrarySelectors(lib);
      showNotification(
        `本地库已刷新：${lib.skills.items.length} 个角色 · ${lib.engines.items.length} 个引擎 · ${lib.models.models.length} 个模型`,
        "success"
      );
    } catch (err) {
      showNotification(`刷新失败: ${err.message}`, "error");
    } finally {
      el("btn-rescan-library").disabled = false;
    }
  }

  async function handleLoadSkill(explicitPath = null) {
    const pathVal = (explicitPath !== null ? explicitPath : el("skill-path-input").value).trim();
    if (!pathVal) {
      showNotification("请先选择角色或输入 Skill 路径", "error");
      return;
    }
    const excludeMeta = el("skill-exclude-meta").checked;
    const maxChars = parseInt(el("skill-max-chars").value || "0", 10);
    el("btn-load-skill").disabled = true;
    try {
      await apiRequest("PUT", "/api/config", {
        skill: { exclude_meta_docs: excludeMeta, max_prompt_chars: maxChars },
      });
      const res = await apiRequest("POST", "/api/skill/load", {
        path: pathVal,
        exclude_meta_docs: excludeMeta,
        max_prompt_chars: maxChars,
      });
      renderSkillSummary(res);
      showNotification(`已切换角色：${res.character_name}`, "success");
    } catch (err) {
      showNotification(`加载角色失败: ${err.message}`, "error");
    } finally {
      el("btn-load-skill").disabled = false;
    }
  }

  async function handleSaveConfig() {
    try {
      const patch = collectConfigPatch();
      const updated = await apiRequest("PUT", "/api/config", patch);
      populateConfigForm(updated);
      const [t2i, lib] = await Promise.all([
        apiRequest("GET", "/api/extensions/t2i"),
        apiRequest("GET", "/api/library"),
      ]);
      renderT2IStatus(t2i);
      renderLibrarySelectors(lib);
      showNotification("配置已保存", "success");
      closeModal("settings-modal");
    } catch (err) {
      showNotification(`保存配置失败: ${err.message}`, "error");
    }
  }

  async function handleStartEngine() {
    el("btn-start-engine").disabled = true;
    try {
      const patch = collectConfigPatch();
      await apiRequest("PUT", "/api/config", patch);
      renderEngineStatus({ state: "starting", mode: patch.llama.engine_mode });
      try {
        const st = await apiRequest("POST", "/api/engine/start", {});
        renderEngineStatus(st);
        showNotification("llama.cpp 引擎已就绪", "success");
      } catch (startErr) {
        if (String(startErr.message || "").includes("already in use")) {
          const res = await apiRequest("POST", "/api/quick-ready", {});
          if (res.config) populateConfigForm(res.config);
          if (res.engine) renderEngineStatus(res.engine);
          showNotification(res.notice || "llama.cpp 引擎已就绪", "success");
        } else {
          throw startErr;
        }
      }
    } catch (err) {
      refreshEngineStatusQuietly();
      showNotification(`引擎启动失败: ${err.message}`, "error");
    } finally {
      el("btn-start-engine").disabled = false;
    }
  }

  async function handleStopEngine() {
    el("btn-stop-engine").disabled = true;
    try {
      const st = await apiRequest("POST", "/api/engine/stop", {});
      renderEngineStatus(st);
      showNotification("引擎已停止", "info");
    } catch (err) {
      showNotification(`停止引擎失败: ${err.message}`, "error");
    } finally {
      el("btn-stop-engine").disabled = false;
    }
  }

  async function handleToggleImmersion() {
    const next = !state.immersionMode;
    try {
      const updated = await apiRequest("PUT", "/api/config", {
        skill: { immersion_mode: next },
      });
      populateConfigForm(updated);
      showNotification(next ? "已开启深度沉浸扮演模式" : "已关闭沉浸扮演增强指令", "success");
    } catch (err) {
      showNotification(`切换失败: ${err.message}`, "error");
    }
  }

  async function handleSaveQuickDirector() {
    const scenario = el("quick-scenario").value.trim();
    const authorsNote = el("quick-authors-note").value.trim();
    try {
      const updated = await apiRequest("PUT", "/api/config", {
        skill: { scenario, authors_note: authorsNote },
      });
      populateConfigForm(updated);
      showNotification("场景与导演提示已生效", "success");
      el("director-drawer").classList.add("hidden");
    } catch (err) {
      showNotification(`保存失败: ${err.message}`, "error");
    }
  }

  function autoResizeTextarea(ta) {
    ta.style.height = "auto";
    ta.style.height = Math.min(180, Math.max(44, ta.scrollHeight)) + "px";
  }

  // =========================================================================
  // Event Binding & Boot
  // =========================================================================
  document.addEventListener("DOMContentLoaded", () => {
    loadThemePrefs();

    // Sidebar toggle
    el("btn-toggle-sidebar").addEventListener("click", () => {
      el("app-shell").classList.toggle("sidebar-collapsed");
    });

    // Modals & Tabs
    el("btn-open-settings").addEventListener("click", () => openModal("settings-modal", "tab-roleplay"));
    el("btn-welcome-settings").addEventListener("click", () => openModal("settings-modal", "tab-engine"));
    el("btn-open-theme").addEventListener("click", () => openModal("theme-modal"));

    document.querySelectorAll("[data-close-modal]").forEach((btn) => {
      btn.addEventListener("click", () => closeModal(btn.dataset.closeModal));
    });
    document.querySelectorAll(".modal-backdrop").forEach((backdrop) => {
      backdrop.addEventListener("click", (ev) => {
        if (ev.target === backdrop) backdrop.classList.add("hidden");
      });
    });
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => switchSettingsTab(btn.dataset.tab));
    });

    // Theme customizer controls
    document.querySelectorAll(".theme-preset-card").forEach((card) => {
      card.addEventListener("click", () => {
        const preset = card.dataset.preset;
        const pDef = PRESET_DEFAULTS[preset] || PRESET_DEFAULTS.cafe;
        state.theme = {
          ...state.theme,
          preset,
          customColors: false,
          accent: pDef.accent,
          bg: pDef.bg,
          surface: pDef.surface,
        };
        applyTheme(state.theme, true);
      });
    });
    ["theme-accent-color", "theme-bg-color", "theme-surface-color"].forEach((id) => {
      el(id).addEventListener("input", () => {
        state.theme.customColors = true;
        state.theme.accent = el("theme-accent-color").value;
        state.theme.bg = el("theme-bg-color").value;
        state.theme.surface = el("theme-surface-color").value;
        applyTheme(state.theme, true);
      });
    });
    el("theme-font-size").addEventListener("input", (ev) => {
      state.theme.fontSize = parseInt(ev.target.value, 10) || 15;
      applyTheme(state.theme, true);
    });
    el("theme-chat-width").addEventListener("change", (ev) => {
      state.theme.chatWidth = ev.target.value;
      applyTheme(state.theme, true);
    });
    el("theme-layout-style").addEventListener("change", (ev) => {
      state.theme.layout = ev.target.value;
      applyTheme(state.theme, true);
    });
    el("theme-font-family").addEventListener("change", (ev) => {
      state.theme.font = ev.target.value;
      applyTheme(state.theme, true);
    });
    el("btn-reset-theme").addEventListener("click", () => {
      state.theme = { ...DEFAULT_THEME_PREFS };
      applyTheme(state.theme, true);
      showNotification("已恢复默认主题", "info");
    });

    // Sampling presets
    const SAMPLING_PRESETS = {
      balanced: { temperature: 0.8, top_p: 0.9, min_p: 0.05, repeat_penalty: 1.12, presence_penalty: 0.1 },
      creative: { temperature: 0.95, top_p: 0.94, min_p: 0.04, repeat_penalty: 1.15, presence_penalty: 0.2 },
      faithful: { temperature: 0.55, top_p: 0.85, min_p: 0.08, repeat_penalty: 1.08, presence_penalty: 0.0 },
    };
    document.querySelectorAll(".sampling-preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = SAMPLING_PRESETS[btn.dataset.preset];
        if (!p) return;
        el("llama-temperature").value = String(p.temperature);
        el("llama-top-p").value = String(p.top_p);
        el("llama-min-p").value = String(p.min_p);
        el("llama-repeat-penalty").value = String(p.repeat_penalty);
        el("llama-presence-penalty").value = String(p.presence_penalty);
        showNotification(`已应用「${btn.textContent}」采样参数预设`, "info");
      });
    });

    // Quick director drawer & immersion pill
    el("btn-toggle-immersion").addEventListener("click", handleToggleImmersion);
    el("btn-open-director").addEventListener("click", () => {
      el("director-drawer").classList.toggle("hidden");
    });
    el("btn-close-director").addEventListener("click", () => {
      el("director-drawer").classList.add("hidden");
    });
    el("btn-save-director").addEventListener("click", handleSaveQuickDirector);

    // Starter prompt chips
    document.querySelectorAll(".starter-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const prompt = chip.dataset.prompt;
        if (prompt) handleSendChat(prompt);
      });
    });

    // Library & Settings bindings
    el("btn-rescan-library").addEventListener("click", handleRescanLibrary);
    el("btn-new-session").addEventListener("click", handleNewSession);
    el("btn-quick-toggle-engine").addEventListener("click", handleQuickReady);
    el("btn-welcome-quick-ready").addEventListener("click", handleQuickReady);
    el("btn-export-chat").addEventListener("click", handleExportMarkdown);
    el("btn-reset-session").addEventListener("click", handleResetSession);

    el("llama-engine-mode").addEventListener("change", updateEngineModeVisibility);
    el("llama-multimodal-enabled").addEventListener("change", updateMultimodalBadge);
    el("llama-mmproj-path").addEventListener("input", updateMultimodalBadge);

    el("skill-library-select").addEventListener("change", (ev) => {
      if (ev.target.value) {
        el("skill-path-input").value = ev.target.value;
        handleLoadSkill(ev.target.value);
      }
    });
    el("engine-library-select").addEventListener("change", (ev) => {
      if (ev.target.value) el("llama-cpp-path").value = ev.target.value;
    });
    el("model-library-select").addEventListener("change", (ev) => {
      const selOpt = ev.target.selectedOptions && ev.target.selectedOptions[0];
      if (ev.target.value) {
        el("llama-model-path").value = ev.target.value;
        const suggested = selOpt ? selOpt.dataset.suggestedMmproj : "";
        if (suggested) {
          el("llama-mmproj-path").value = suggested;
          el("mmproj-library-select").value = suggested;
        }
        updateMultimodalBadge();
        updateEngineMiniModelText();
      }
    });
    el("mmproj-library-select").addEventListener("change", (ev) => {
      el("llama-mmproj-path").value = ev.target.value || "";
      updateMultimodalBadge();
    });

    el("btn-load-skill").addEventListener("click", () => handleLoadSkill());
    el("btn-save-config").addEventListener("click", handleSaveConfig);
    el("btn-start-engine").addEventListener("click", handleStartEngine);
    el("btn-stop-engine").addEventListener("click", handleStopEngine);

    // Multimodal image upload, paste, drag-drop
    el("btn-add-image").addEventListener("click", () => el("chat-image-input").click());
    el("chat-image-input").addEventListener("change", (ev) => {
      addImageFiles(ev.target.files);
      ev.target.value = "";
    });
    const chatInput = el("chat-input");
    chatInput.addEventListener("paste", (ev) => {
      if (ev.clipboardData && ev.clipboardData.files && ev.clipboardData.files.length > 0) {
        addImageFiles(ev.clipboardData.files);
      }
    });
    chatInput.addEventListener("input", () => autoResizeTextarea(chatInput));
    chatInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        handleSendChat();
      }
    });

    const composerCard = el("composer-card");
    composerCard.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      composerCard.classList.add("drag-over");
    });
    composerCard.addEventListener("dragleave", () => composerCard.classList.remove("drag-over"));
    composerCard.addEventListener("drop", (ev) => {
      ev.preventDefault();
      composerCard.classList.remove("drag-over");
      if (ev.dataTransfer && ev.dataTransfer.files) addImageFiles(ev.dataTransfer.files);
    });

    el("btn-send-chat").addEventListener("click", () => handleSendChat());
    el("btn-stop-stream").addEventListener("click", () => {
      if (state.abortController) state.abortController.abort();
    });
    el("btn-regenerate-last").addEventListener("click", handleRegenerate);

    // Escape closes open modals
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        ["settings-modal", "theme-modal", "lightbox-modal"].forEach(closeModal);
      }
    });

    refreshAllState();
  });
})();
