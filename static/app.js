/* Vanilla ES6 Frontend Controller for Moris Roleplay (with Dedicated Libraries & Multimodal Vision) */
(() => {
  "use strict";

  const state = {
    sessionId: "session-" + Math.random().toString(36).slice(2, 10),
    turnIndex: 0,
    characterName: "角色",
    busy: false,
    pendingImages: [],
    library: {
      engines: { items: [] },
      models: { models: [], mmprojs: [] },
      skills: { items: [] },
    },
  };

  const el = (id) => document.getElementById(id);

  function showNotification(message, kind = "error") {
    const banner = el("notification-banner");
    if (!banner) return;
    banner.textContent = message;
    banner.className = `notification-banner ${kind}`;
    banner.classList.remove("hidden");
    setTimeout(() => {
      banner.classList.add("hidden");
    }, 6000);
  }

  async function apiRequest(method, path, body = undefined) {
    const opts = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (body !== undefined) {
      opts.body = JSON.stringify(body);
    }
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

  function setBadge(badgeEl, text, variant) {
    if (!badgeEl) return;
    badgeEl.textContent = text;
    badgeEl.className = `badge badge-${variant}`;
  }

  function updateEngineModeVisibility() {
    const mode = el("llama-engine-mode").value;
    const managedFields = el("managed-fields");
    const externalFields = el("external-fields");
    if (mode === "external_server") {
      managedFields.classList.add("hidden");
      externalFields.classList.remove("hidden");
    } else {
      managedFields.classList.remove("hidden");
      externalFields.classList.add("hidden");
    }
    updateMultimodalBadge();
  }

  function updateMultimodalBadge() {
    const mmPill = el("multimodal-mode-pill");
    const addImgBtn = el("btn-add-image");
    const mmEnabled = el("llama-multimodal-enabled").checked;
    const mmprojPath = el("llama-mmproj-path").value.trim();
    const mode = el("llama-engine-mode").value;

    if (!mmEnabled) {
      mmPill.textContent = "多模态: 已关闭";
      addImgBtn.disabled = true;
      return;
    }
    addImgBtn.disabled = false;
    if (mmprojPath) {
      mmPill.textContent = "多模态: 视觉已绑定 (mmproj)";
    } else if (mode === "external_server") {
      mmPill.textContent = "多模态: 外部服务就绪";
    } else {
      mmPill.textContent = "多模态: 纯文本 (可绑定 mmproj)";
    }
  }

  function renderLibrarySelectors(libData) {
    if (!libData) return;
    state.library = libData;

    // 1. Skills Library Dropdown
    const skillsList = (libData.skills && libData.skills.items) || [];
    el("skill-lib-count").textContent = String(skillsList.length);
    const skillSelect = el("skill-library-select");
    const currentSkillPath = el("skill-path-input").value.trim();
    skillSelect.innerHTML =
      '<option value="">-- 请从 Skill 库选择角色，或在下方输入自定义路径 --</option>';
    skillsList.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.path;
      opt.textContent = `[${item.library_label}] ${item.display_name} (${item.md_file_count} .md)`;
      skillSelect.appendChild(opt);
    });
    if (currentSkillPath && [...skillSelect.options].some((o) => o.value === currentSkillPath)) {
      skillSelect.value = currentSkillPath;
    }

    // 2. Engine Library Dropdown
    const engineItems = (libData.engines && libData.engines.items) || [];
    el("engine-lib-count").textContent = String(engineItems.length);
    const engSelect = el("engine-library-select");
    const currentEngPath = el("llama-cpp-path").value.trim();
    engSelect.innerHTML =
      '<option value="">-- 从 engines/ 库选择或手动填写下方路径 --</option>';
    engineItems.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.path;
      const mmTag = item.supports_multimodal ? "支持多模态" : "CLI";
      opt.textContent = `${item.name} [${item.kind} · ${mmTag}] (${item.size_mb} MB)`;
      engSelect.appendChild(opt);
    });
    if (currentEngPath && [...engSelect.options].some((o) => o.value === currentEngPath)) {
      engSelect.value = currentEngPath;
    } else if (!currentEngPath && engineItems.length > 0) {
      engSelect.value = engineItems[0].path;
      el("llama-cpp-path").value = engineItems[0].path;
    }

    // 3. Model Library Dropdown (Main GGUF models)
    const modelItems = (libData.models && libData.models.models) || [];
    el("model-lib-count").textContent = String(modelItems.length);
    const modelSelect = el("model-library-select");
    const currentModelPath = el("llama-model-path").value.trim();
    modelSelect.innerHTML =
      '<option value="">-- 从 models/ 库选择 .gguf 主模型或手动填写 --</option>';
    modelItems.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.path;
      opt.dataset.suggestedMmproj = item.suggested_mmproj || "";
      const modTag = item.modality === "multimodal" ? "多模态视觉" : "纯文本 LLM";
      opt.textContent = `[${modTag}] ${item.rel_path} (${item.size_mb} MB)`;
      modelSelect.appendChild(opt);
    });
    if (currentModelPath && [...modelSelect.options].some((o) => o.value === currentModelPath)) {
      modelSelect.value = currentModelPath;
    }

    // 4. Multimodal Projector (mmproj) Dropdown
    const mmprojItems = (libData.models && libData.models.mmprojs) || [];
    el("mmproj-lib-count").textContent = String(mmprojItems.length);
    const mmprojSelect = el("mmproj-library-select");
    const currentMmprojPath = el("llama-mmproj-path").value.trim();
    mmprojSelect.innerHTML =
      '<option value="">-- 纯文本模式 (不使用 mmproj 视觉投影器) --</option>';
    mmprojItems.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.path;
      opt.textContent = `[mmproj] ${item.rel_path} (${item.size_mb} MB)`;
      mmprojSelect.appendChild(opt);
    });
    if (currentMmprojPath && [...mmprojSelect.options].some((o) => o.value === currentMmprojPath)) {
      mmprojSelect.value = currentMmprojPath;
    }

    updateMultimodalBadge();
  }

  function renderSkillSummary(skillData) {
    const box = el("skill-summary-box");
    const badge = el("skill-status-badge");
    const titleEl = el("active-character-title");
    const descEl = el("active-character-desc");

    if (!skillData || skillData.loaded === false || !skillData.character_name) {
      box.classList.add("hidden");
      setBadge(badge, "未加载", "neutral");
      titleEl.textContent = "未加载角色 Skill（通用对话模式）";
      descEl.textContent =
        "请先在左侧从 Skill 库选择或输入角色 Skill (.md) 路径并点击“加载角色 Skill”";
      state.characterName = "Assistant";
      return;
    }

    state.characterName = skillData.character_name;
    setBadge(badge, "已加载", "success");
    box.classList.remove("hidden");

    el("skill-char-name").textContent = skillData.character_name;
    el("skill-source-type").textContent =
      skillData.source_type === "directory" ? "文件夹 (.md)" : "单文件 (.md)";
    el("skill-stats").textContent = `${skillData.total_chars || 0} chars / ~${
      skillData.estimated_tokens || 0
    } tokens`;

    const loadedFiles = skillData.loaded_files || [];
    el("skill-loaded-count").textContent = String(loadedFiles.length);
    const loadedContainer = el("skill-loaded-files");
    loadedContainer.innerHTML = "";
    loadedFiles.forEach((f) => {
      const span = document.createElement("span");
      span.className = "file-pill";
      span.textContent = f;
      loadedContainer.appendChild(span);
    });

    const ignoredFiles = skillData.ignored_files || [];
    el("skill-ignored-count").textContent = String(ignoredFiles.length);
    const ignoredContainer = el("skill-ignored-files");
    ignoredContainer.innerHTML = "";
    ignoredFiles.forEach((f) => {
      const span = document.createElement("span");
      span.className = "file-pill";
      span.textContent = f;
      ignoredContainer.appendChild(span);
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
    titleEl.textContent = `当前扮演角色：${skillData.character_name}`;
    descEl.textContent =
      skillData.description ||
      `已从 ${skillData.source_path} 加载 ${loadedFiles.length} 个 .md 设定文件`;
  }

  function renderEngineStatus(eng) {
    if (!eng) return;
    const badge = el("engine-status-badge");
    const st = eng.state || "stopped";
    let variant = "neutral";
    if (st === "ready") variant = "success";
    else if (st === "starting") variant = "warning";
    else if (st === "error") variant = "danger";
    setBadge(badge, st, variant);

    const parts = [];
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
      setBadge(badge, `已启用 (${t2iSummary.active_plugin})`, "success");
    } else {
      setBadge(badge, "未启用", "neutral");
    }

    const select = el("t2i-active-plugin");
    const currentVal = t2iSummary.active_plugin || select.value || "noop";
    const plugins = t2iSummary.plugins || [];
    if (plugins.length > 0) {
      select.innerHTML = "";
      plugins.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.description ? `${p.name} — ${p.description}` : p.name;
        select.appendChild(opt);
      });
      if (![...select.options].some((o) => o.value === currentVal)) {
        const extraOpt = document.createElement("option");
        extraOpt.value = currentVal;
        extraOpt.textContent = currentVal;
        select.appendChild(extraOpt);
      }
      select.value = currentVal;
    }
  }

  function populateConfigForm(cfg) {
    if (!cfg) return;
    if (cfg.skill) {
      if (cfg.skill.last_path && !el("skill-path-input").value) {
        el("skill-path-input").value = cfg.skill.last_path;
      }
      el("skill-exclude-meta").checked = Boolean(cfg.skill.exclude_meta_docs);
      el("skill-max-chars").value = String(cfg.skill.max_prompt_chars ?? 0);
    }
    if (cfg.llama) {
      el("llama-engine-mode").value = cfg.llama.engine_mode || "managed_server";
      el("llama-cpp-path").value = cfg.llama.llama_cpp_path || "";
      el("llama-model-path").value = cfg.llama.model_path || "";
      el("llama-mmproj-path").value = cfg.llama.mmproj_path || "";
      el("llama-multimodal-enabled").checked =
        cfg.llama.multimodal_enabled !== undefined ? Boolean(cfg.llama.multimodal_enabled) : true;
      el("llama-server-url").value = cfg.llama.server_url || "http://127.0.0.1:8080";
      el("llama-port").value = String(cfg.llama.port ?? 8080);
      el("llama-ctx-size").value = String(cfg.llama.ctx_size ?? 16384);
      el("llama-ngl").value = String(cfg.llama.n_gpu_layers ?? -1);
      el("llama-temperature").value = String(cfg.llama.temperature ?? 0.8);
      el("llama-top-p").value = String(cfg.llama.top_p ?? 0.9);
      el("llama-max-tokens").value = String(cfg.llama.max_tokens ?? 512);
      el("llama-max-history").value = String(cfg.llama.max_history_turns ?? 20);
      updateEngineModeVisibility();
    }
    if (cfg.t2i) {
      el("t2i-enabled").checked = Boolean(cfg.t2i.enabled);
      el("t2i-context-turns").value = String(cfg.t2i.context_window_turns ?? 4);
      el("t2i-plugin-settings").value = JSON.stringify(
        cfg.t2i.plugin_settings || {},
        null,
        2
      );
    }
  }

  function collectConfigPatch() {
    let pluginSettings = {};
    const rawSettings = el("t2i-plugin-settings").value.trim();
    if (rawSettings) {
      pluginSettings = JSON.parse(rawSettings);
    }
    return {
      skill: {
        last_path: el("skill-path-input").value.trim(),
        exclude_meta_docs: el("skill-exclude-meta").checked,
        max_prompt_chars: parseInt(el("skill-max-chars").value || "0", 10),
      },
      llama: {
        engine_mode: el("llama-engine-mode").value,
        llama_cpp_path: el("llama-cpp-path").value.trim(),
        model_path: el("llama-model-path").value.trim(),
        mmproj_path: el("llama-mmproj-path").value.trim(),
        multimodal_enabled: el("llama-multimodal-enabled").checked,
        server_url: el("llama-server-url").value.trim(),
        port: parseInt(el("llama-port").value || "8080", 10),
        ctx_size: parseInt(el("llama-ctx-size").value || "16384", 10),
        n_gpu_layers: parseInt(el("llama-ngl").value || "-1", 10),
        temperature: parseFloat(el("llama-temperature").value || "0.8"),
        top_p: parseFloat(el("llama-top-p").value || "0.9"),
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
      showNotification("请先勾选“启用视觉多模态图片输入”", "error");
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

  function appendChatMessage(role, author, text, turnIndex, t2iHook = null, images = []) {
    const placeholder = el("empty-chat-placeholder");
    if (placeholder) placeholder.classList.add("hidden");

    const container = el("chat-messages");
    const item = document.createElement("div");
    item.className = `message-item ${role === "user" ? "message-user" : "message-assistant"}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    const authorSpan = document.createElement("strong");
    authorSpan.textContent = author;
    const turnSpan = document.createElement("span");
    turnSpan.textContent = `Turn #${turnIndex}`;
    meta.appendChild(authorSpan);
    meta.appendChild(turnSpan);

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = text;

    item.appendChild(meta);
    item.appendChild(content);

    if (images && images.length > 0) {
      const imgRow = document.createElement("div");
      imgRow.className = "message-images";
      images.forEach((src) => {
        const imgEl = document.createElement("img");
        imgEl.src = src;
        imgEl.alt = "用户发送的多模态图片";
        imgRow.appendChild(imgEl);
      });
      item.appendChild(imgRow);
    }

    if (t2iHook && t2iHook.invoked) {
      const hookCard = document.createElement("div");
      hookCard.className = "t2i-hook-card";
      hookCard.textContent = `[场景文生图 Hook · ${t2iHook.plugin_name}] 状态: ${
        t2iHook.status
      }${t2iHook.prompt_used ? " | 场景提示词: " + t2iHook.prompt_used : ""}`;
      item.appendChild(hookCard);
    }

    container.appendChild(item);
    container.scrollTop = container.scrollHeight;
  }

  async function refreshAllState() {
    try {
      const [health, cfg, t2i, lib] = await Promise.all([
        apiRequest("GET", "/api/health"),
        apiRequest("GET", "/api/config"),
        apiRequest("GET", "/api/extensions/t2i"),
        apiRequest("GET", "/api/library"),
      ]);
      setBadge(el("health-badge"), "在线", "success");
      populateConfigForm(cfg);
      renderLibrarySelectors(lib);
      renderSkillSummary(health.skill);
      renderEngineStatus(health.engine);
      renderT2IStatus(t2i);
    } catch (err) {
      setBadge(el("health-badge"), "离线", "danger");
      showNotification(`无法获取服务状态: ${err.message}`, "error");
    }
  }

  async function handleRescanLibrary() {
    el("btn-rescan-library").disabled = true;
    try {
      const lib = await apiRequest("POST", "/api/library/rescan", {});
      renderLibrarySelectors(lib);
      showNotification(
        `本地库刷新完成：发现 ${lib.skills.items.length} 个角色 Skill、${lib.engines.items.length} 个引擎、${lib.models.models.length} 个主模型`,
        "success"
      );
    } catch (err) {
      showNotification(`刷新本地库失败: ${err.message}`, "error");
    } finally {
      el("btn-rescan-library").disabled = false;
    }
  }

  async function handleLoadSkill() {
    const pathVal = el("skill-path-input").value.trim();
    if (!pathVal) {
      showNotification("请先从 Skill 库选择或输入角色 Skill (.md) 路径", "error");
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
      showNotification(
        `已成功加载角色 Skill: ${res.character_name} (${res.loaded_files.length} 个 .md 文件)`,
        "success"
      );
    } catch (err) {
      showNotification(`加载角色 Skill 失败: ${err.message}`, "error");
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
    } catch (err) {
      showNotification(`保存配置失败: ${err.message}`, "error");
    }
  }

  async function handleStartEngine() {
    el("btn-start-engine").disabled = true;
    try {
      const patch = collectConfigPatch();
      await apiRequest("PUT", "/api/config", patch);
      setBadge(el("engine-status-badge"), "starting", "warning");
      const st = await apiRequest("POST", "/api/engine/start", {});
      renderEngineStatus(st);
      showNotification("llama.cpp 引擎已就绪", "success");
    } catch (err) {
      try {
        const st = await apiRequest("GET", "/api/engine/status");
        renderEngineStatus(st);
      } catch (_ignore) {}
      showNotification(`引擎启动/连接失败: ${err.message}`, "error");
    } finally {
      el("btn-start-engine").disabled = false;
    }
  }

  async function handleStopEngine() {
    el("btn-stop-engine").disabled = true;
    try {
      const st = await apiRequest("POST", "/api/engine/stop", {});
      renderEngineStatus(st);
      showNotification("引擎已停止", "success");
    } catch (err) {
      showNotification(`停止引擎失败: ${err.message}`, "error");
    } finally {
      el("btn-stop-engine").disabled = false;
    }
  }

  async function handleSendChat() {
    if (state.busy) return;
    const inputEl = el("chat-input");
    const msg = inputEl.value.trim();
    if (!msg) return;

    const imagesToSend = state.pendingImages.slice();
    state.pendingImages = [];
    renderPendingImages();

    state.busy = true;
    el("btn-send-chat").disabled = true;
    const optimisticTurn = state.turnIndex + 1;
    appendChatMessage("user", "你 (User)", msg, optimisticTurn, null, imagesToSend);
    inputEl.value = "";

    const payload = {
      session_id: state.sessionId,
      message: msg,
    };
    if (imagesToSend.length > 0) {
      payload.images = imagesToSend;
    }

    try {
      const res = await apiRequest("POST", "/api/chat", payload);
      state.turnIndex = res.turn_index;
      el("turn-counter-label").textContent = `当前轮次: ${res.turn_index} (消息数: ${res.messages_count})`;
      appendChatMessage(
        "assistant",
        state.characterName || "Character",
        res.reply,
        res.turn_index,
        res.t2i_hook
      );

      if (res.t2i_hook) {
        el("t2i-last-status").textContent = `${res.t2i_hook.status} (plugin: ${res.t2i_hook.plugin_name}, invoked: ${res.t2i_hook.invoked})`;
        el("t2i-last-prompt").textContent =
          res.t2i_hook.prompt_used || res.t2i_hook.message || "";
      }
    } catch (err) {
      showNotification(`发送对话失败: ${err.message}`, "error");
    } finally {
      state.busy = false;
      el("btn-send-chat").disabled = false;
      inputEl.focus();
    }
  }

  async function handleResetSession() {
    try {
      await apiRequest("POST", "/api/sessions/reset", { session_id: state.sessionId });
      state.turnIndex = 0;
      state.pendingImages = [];
      renderPendingImages();
      el("turn-counter-label").textContent = "当前轮次: 0";
      const container = el("chat-messages");
      container.innerHTML = "";
      showNotification("当前会话历史已清空（已保留当前加载的角色 Skill）", "success");
    } catch (err) {
      showNotification(`清空会话失败: ${err.message}`, "error");
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    el("session-id-label").textContent = state.sessionId;
    el("llama-engine-mode").addEventListener("change", updateEngineModeVisibility);
    el("llama-multimodal-enabled").addEventListener("change", updateMultimodalBadge);
    el("llama-mmproj-path").addEventListener("input", updateMultimodalBadge);
    el("btn-rescan-library").addEventListener("click", handleRescanLibrary);

    // Skill Library dropdown auto-fills path and triggers load
    el("skill-library-select").addEventListener("change", (ev) => {
      const selected = ev.target.value;
      if (selected) {
        el("skill-path-input").value = selected;
        handleLoadSkill();
      }
    });

    // Engine Library dropdown auto-fills llama_cpp_path
    el("engine-library-select").addEventListener("change", (ev) => {
      if (ev.target.value) {
        el("llama-cpp-path").value = ev.target.value;
      }
    });

    // Model Library dropdown auto-fills model_path and auto-pairs suggested mmproj
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
      }
    });

    // mmproj Library dropdown auto-fills mmproj_path
    el("mmproj-library-select").addEventListener("change", (ev) => {
      el("llama-mmproj-path").value = ev.target.value || "";
      updateMultimodalBadge();
    });

    // Multimodal image upload & clipboard paste
    el("btn-add-image").addEventListener("click", () => {
      el("chat-image-input").click();
    });
    el("chat-image-input").addEventListener("change", (ev) => {
      addImageFiles(ev.target.files);
      ev.target.value = "";
    });
    el("chat-input").addEventListener("paste", (ev) => {
      if (ev.clipboardData && ev.clipboardData.files && ev.clipboardData.files.length > 0) {
        addImageFiles(ev.clipboardData.files);
      }
    });

    el("btn-load-skill").addEventListener("click", handleLoadSkill);
    el("btn-save-config").addEventListener("click", handleSaveConfig);
    el("btn-save-t2i").addEventListener("click", handleSaveConfig);
    el("btn-start-engine").addEventListener("click", handleStartEngine);
    el("btn-stop-engine").addEventListener("click", handleStopEngine);
    el("btn-send-chat").addEventListener("click", handleSendChat);
    el("btn-reset-session").addEventListener("click", handleResetSession);

    el("chat-input").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        handleSendChat();
      }
    });

    refreshAllState();
  });
})();
