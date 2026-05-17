(function () {
  "use strict";

  const API_BASE_KEY = "travelAssistant.apiBase";
  const DEFAULT_API_BASE = "http://127.0.0.1:8000";

  const STEPS = [
    { key: "router", title: "理解需求" },
    { key: "researcher", title: "查找资料" },
    { key: "planner", title: "整理路线" },
    { key: "ticket_agent", title: "车票信息" },
  ];

  const STATUS_TEXT = {
    pending: "待开始",
    running: "处理中",
    completed: "已完成",
    skipped: "本次跳过",
    failed: "出错了",
  };

  const INTENT_TEXT = {
    need_plan: "已判断为行程规划",
    need_answer: "已判断为旅行问答",
    need_ticket: "已判断为车票查询",
    missing_info: "还需要补充部分信息",
    other: "已收到需求",
  };

  const state = {
    apiBase: DEFAULT_API_BASE,
    activeSessionId: null,
    activeSession: null,
    sessions: [],
    selectedFiles: [],
    isBusy: false,
    toastTimer: null,
  };

  const els = {
    apiBaseInput: document.getElementById("apiBaseInput"),
    saveApiBaseBtn: document.getElementById("saveApiBaseBtn"),
    healthStatus: document.getElementById("healthStatus"),
    sessionList: document.getElementById("sessionList"),
    newSessionBtn: document.getElementById("newSessionBtn"),
    activeSessionTitle: document.getElementById("activeSessionTitle"),
    activeSessionMeta: document.getElementById("activeSessionMeta"),
    modelSelect: document.getElementById("modelSelect"),
    messageList: document.getElementById("messageList"),
    chatForm: document.getElementById("chatForm"),
    messageInput: document.getElementById("messageInput"),
    sendBtn: document.getElementById("sendBtn"),
    attachBtn: document.getElementById("attachBtn"),
    chatFileInput: document.getElementById("chatFileInput"),
    selectedFiles: document.getElementById("selectedFiles"),
    runtimeList: document.getElementById("runtimeList"),
    elapsedText: document.getElementById("elapsedText"),
    clearSessionBtn: document.getElementById("clearSessionBtn"),
    kbBadge: document.getElementById("kbBadge"),
    kbStatusText: document.getElementById("kbStatusText"),
    kbFileInput: document.getElementById("kbFileInput"),
    uploadKbBtn: document.getElementById("uploadKbBtn"),
    clearKbBtn: document.getElementById("clearKbBtn"),
    toast: document.getElementById("toast"),
    sidebar: document.querySelector(".sidebar"),
    sidebarToggleBtn: document.getElementById("sidebarToggleBtn"),
  };

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    state.apiBase = normalizeApiBase(localStorage.getItem(API_BASE_KEY) || DEFAULT_API_BASE);
    els.apiBaseInput.value = state.apiBase;

    bindEvents();
    renderRuntime();
    renderSelectedFiles();
    setBusy(false);

    await Promise.allSettled([loadModels(), loadSessions(), refreshHealth(), refreshKnowledgeStatus()]);
  }

  function bindEvents() {
    els.saveApiBaseBtn.addEventListener("click", async () => {
      state.apiBase = normalizeApiBase(els.apiBaseInput.value || DEFAULT_API_BASE);
      els.apiBaseInput.value = state.apiBase;
      localStorage.setItem(API_BASE_KEY, state.apiBase);
      showToast("后端地址已保存");
      await Promise.allSettled([loadModels(), loadSessions(), refreshHealth(), refreshKnowledgeStatus()]);
    });

    els.newSessionBtn.addEventListener("click", createSession);
    els.clearSessionBtn.addEventListener("click", clearActiveSession);
    els.attachBtn.addEventListener("click", () => els.chatFileInput.click());
    els.chatFileInput.addEventListener("change", handleChatFilesSelected);
    els.chatForm.addEventListener("submit", handleSubmit);

    els.messageInput.addEventListener("input", () => {
      resizeTextarea(els.messageInput);
    });

    els.messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        els.chatForm.requestSubmit();
      }
    });

    els.uploadKbBtn.addEventListener("click", () => els.kbFileInput.click());
    els.kbFileInput.addEventListener("change", uploadKnowledgeFiles);
    els.clearKbBtn.addEventListener("click", clearKnowledgeBase);

    document.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => {
        els.messageInput.value = button.getAttribute("data-prompt") || "";
        resizeTextarea(els.messageInput);
        els.messageInput.focus();
      });
    });

    els.sidebarToggleBtn.addEventListener("click", () => {
      els.sidebar.classList.toggle("open");
    });

    document.addEventListener("click", (event) => {
      if (!els.sidebar.classList.contains("open")) {
        return;
      }
      const clickedInside = els.sidebar.contains(event.target) || els.sidebarToggleBtn.contains(event.target);
      if (!clickedInside) {
        els.sidebar.classList.remove("open");
      }
    });
  }

  function normalizeApiBase(value) {
    return String(value || DEFAULT_API_BASE).trim().replace(/\/+$/, "") || DEFAULT_API_BASE;
  }

  function apiUrl(path) {
    return `${state.apiBase}${path}`;
  }

  async function fetchJson(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const isForm = options.body instanceof FormData;
    if (!isForm && options.body !== undefined && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    const response = await fetch(apiUrl(path), { ...options, headers });
    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }
    return response.json();
  }

  async function readErrorMessage(response) {
    const clone = response.clone();
    try {
      const payload = await response.json();
      return payload.detail || payload.message || `请求失败：${response.status}`;
    } catch (error) {
      const text = await clone.text().catch(() => "");
      return text || `请求失败：${response.status}`;
    }
  }

  async function refreshHealth() {
    setHealth("checking", "正在连接");
    try {
      const payload = await fetchJson("/api/health");
      const chunks = payload.knowledge_base?.chunk_count ?? 0;
      setHealth("ok", `服务在线 · 资料 ${chunks} 条`);
    } catch (error) {
      setHealth("error", `连接失败：${error.message}`);
    }
  }

  function setHealth(type, text) {
    els.healthStatus.className = "status-pill";
    if (type === "ok") {
      els.healthStatus.classList.add("status-ok");
    } else if (type === "error") {
      els.healthStatus.classList.add("status-error");
    } else {
      els.healthStatus.classList.add("status-muted");
    }
    els.healthStatus.textContent = text;
  }

  async function loadModels() {
    try {
      const payload = await fetchJson("/api/models");
      const models = Array.isArray(payload.models) ? payload.models : [];
      const defaultModel = payload.default_model || models[0] || "glm-4.5-air";

      els.modelSelect.innerHTML = "";
      [...new Set([defaultModel, ...models])].forEach((model) => {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        els.modelSelect.appendChild(option);
      });
      els.modelSelect.value = defaultModel;
    } catch (error) {
      els.modelSelect.innerHTML = '<option value="glm-4.5-air">glm-4.5-air</option>';
      showToast(`模型列表加载失败：${error.message}`);
    }
  }

  async function loadSessions(preferredSessionId = state.activeSessionId, options = {}) {
    try {
      const payload = await fetchJson("/api/sessions");
      state.sessions = Array.isArray(payload.sessions) ? payload.sessions : [];
      state.sessions.sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));

      const preferredExists = state.sessions.some((session) => session.id === preferredSessionId);
      const nextSessionId = preferredExists ? preferredSessionId : state.sessions[0]?.id || null;

      renderSessions();

      if (nextSessionId) {
        await loadSession(nextSessionId, options);
      } else {
        state.activeSessionId = null;
        state.activeSession = null;
        renderMessages([]);
        updateSessionHeader();
      }
    } catch (error) {
      showToast(`会话加载失败：${error.message}`);
      renderSessions();
    }
  }

  async function refreshSessionSummaries() {
    const payload = await fetchJson("/api/sessions");
    state.sessions = Array.isArray(payload.sessions) ? payload.sessions : [];
    state.sessions.sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));

    const activeSummary = state.sessions.find((session) => session.id === state.activeSessionId);
    if (activeSummary) {
      state.activeSession = {
        ...(state.activeSession || {}),
        ...activeSummary,
        messages: state.activeSession?.messages || [],
      };
      updateSessionHeader();
    }
    renderSessions();
  }

  function renderSessions() {
    if (!state.sessions.length) {
      els.sessionList.innerHTML = '<div class="muted-text">暂无会话</div>';
      return;
    }

    els.sessionList.innerHTML = "";
    state.sessions.forEach((session) => {
      const row = document.createElement("div");
      row.className = `session-item${session.id === state.activeSessionId ? " active" : ""}`;

      const contentButton = document.createElement("button");
      contentButton.type = "button";
      contentButton.className = "session-content-button";
      contentButton.style.all = "unset";
      contentButton.style.minWidth = "0";
      contentButton.style.cursor = "pointer";
      contentButton.innerHTML = `
        <div class="session-title">${escapeHtml(session.title || "新会话")}</div>
        <div class="session-meta">${session.message_count || 0} 条消息 · ${formatDate(session.updated_at)}</div>
      `;
      contentButton.addEventListener("click", async () => {
        await loadSession(session.id);
        els.sidebar.classList.remove("open");
      });

      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "delete-session-btn";
      deleteButton.title = "删除会话";
      deleteButton.setAttribute("aria-label", "删除会话");
      deleteButton.textContent = "×";
      deleteButton.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteSession(session.id);
      });

      row.append(contentButton, deleteButton);
      els.sessionList.appendChild(row);
    });
  }

  async function loadSession(sessionId, options = {}) {
    try {
      const payload = await fetchJson(`/api/sessions/${encodeURIComponent(sessionId)}`);
      state.activeSession = payload.session;
      state.activeSessionId = payload.session.id;
      renderSessions();
      renderMessages(payload.session.messages || []);
      updateSessionHeader();
      if (options.resetRuntime !== false) {
        renderRuntime();
      }
      setBusy(state.isBusy);
    } catch (error) {
      showToast(`会话读取失败：${error.message}`);
    }
  }

  async function createSession() {
    try {
      const payload = await fetchJson("/api/sessions", {
        method: "POST",
        body: JSON.stringify({ title: null }),
      });
      const id = payload.session?.id;
      state.activeSessionId = id || null;
      await loadSessions(id);
      showToast("已创建新会话");
    } catch (error) {
      showToast(`新建会话失败：${error.message}`);
    }
  }

  async function deleteSession(sessionId) {
    if (!window.confirm("确定删除这个会话吗？")) {
      return;
    }

    try {
      const payload = await fetchJson(`/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
      const nextId = payload.current_session || null;
      state.activeSessionId = sessionId === state.activeSessionId ? nextId : state.activeSessionId;
      await loadSessions(state.activeSessionId);
      showToast("会话已删除");
    } catch (error) {
      showToast(`删除会话失败：${error.message}`);
    }
  }

  async function clearActiveSession() {
    if (!state.activeSessionId) {
      return;
    }

    if (!window.confirm("确定清空当前会话消息吗？")) {
      return;
    }

    try {
      const payload = await fetchJson(`/api/sessions/${encodeURIComponent(state.activeSessionId)}/messages`, {
        method: "DELETE",
      });
      state.activeSession = payload.session;
      renderMessages([]);
      updateSessionHeader();
      await loadSessions(state.activeSessionId);
      showToast("当前会话已清空");
    } catch (error) {
      showToast(`清空会话失败：${error.message}`);
    }
  }

  function updateSessionHeader() {
    const session = state.activeSession;
    els.activeSessionTitle.textContent = session?.title || "新会话";
    if (!session) {
      els.activeSessionMeta.textContent = "选择一条记录，继续完善路线";
      return;
    }
    els.activeSessionMeta.textContent = `${session.message_count || 0} 条消息 · 更新于 ${formatDate(session.updated_at)}`;
  }

  function handleChatFilesSelected(event) {
    const files = Array.from(event.target.files || []);
    state.selectedFiles.push(...files);
    event.target.value = "";
    renderSelectedFiles();
  }

  function renderSelectedFiles() {
    els.selectedFiles.innerHTML = "";

    state.selectedFiles.forEach((file, index) => {
      const chip = document.createElement("div");
      chip.className = "file-chip";
      chip.innerHTML = `<span title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>`;

      const removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.title = "移除文件";
      removeButton.setAttribute("aria-label", "移除文件");
      removeButton.textContent = "×";
      removeButton.addEventListener("click", () => {
        state.selectedFiles.splice(index, 1);
        renderSelectedFiles();
      });

      chip.appendChild(removeButton);
      els.selectedFiles.appendChild(chip);
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (state.isBusy) {
      return;
    }

    const message = els.messageInput.value.trim();
    const files = [...state.selectedFiles];
    if (!message && !files.length) {
      showToast("先说一点旅行想法，或上传图片/语音");
      return;
    }

    els.messageInput.value = "";
    resizeTextarea(els.messageInput);
    state.selectedFiles = [];
    renderSelectedFiles();

    const displayText = files.length ? `${message || "已上传文件"}\n\n${files.map((file) => `- ${file.name}`).join("\n")}` : message;
    appendMessage("user", displayText);
    const assistantBubble = appendMessage("assistant", "", true);
    setBusy(true);
    renderRuntime();
    els.elapsedText.textContent = "正在整理路线";

    try {
      if (files.length) {
        await sendChatWithFiles(message, files, assistantBubble);
      } else {
        await sendStreamingChat(message, assistantBubble);
      }
      await Promise.allSettled([refreshSessionSummaries(), refreshHealth(), refreshKnowledgeStatus()]);
    } catch (error) {
      assistantBubble.innerHTML = renderMarkdown(`请求失败：${error.message}`);
      showToast(`请求失败：${error.message}`);
    } finally {
      setBusy(false);
      scrollMessagesToBottom();
    }
  }

  async function sendChatWithFiles(message, files, assistantBubble) {
    const formData = new FormData();
    formData.append("message", message);
    formData.append("model", getSelectedModel());
    formData.append("save_to_session", "true");
    if (state.activeSessionId) {
      formData.append("session_id", state.activeSessionId);
    }
    files.forEach((file) => formData.append("files", file));

    const payload = await fetchJson("/api/chat/files", {
      method: "POST",
      body: formData,
    });

    state.activeSessionId = payload.session_id || state.activeSessionId;
    assistantBubble.innerHTML = renderMarkdown(payload.message || "没有返回可展示内容。");
    renderRuntime(payload.runtime);
    els.elapsedText.textContent = formatElapsed(payload.elapsed);
  }

  async function sendStreamingChat(message, assistantBubble) {
    const response = await fetch(apiUrl("/api/chat/stream"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: state.activeSessionId,
        model: getSelectedModel(),
        save_to_session: true,
      }),
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    if (!response.body) {
      const text = await response.text();
      assistantBubble.innerHTML = renderMarkdown(text || "浏览器不支持流式读取。");
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let finalAnswer = "";
    let sawFinal = false;

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() || "";
      for (const part of parts) {
        const event = parseSseBlock(part);
        if (event) {
          const payload = parseEventPayload(event);
          if (payload) {
            sawFinal = handleStreamEvent(payload, assistantBubble) || sawFinal;
            if (payload.session_id) {
              state.activeSessionId = payload.session_id;
            }
            if (payload.type === "final") {
              finalAnswer = payload.answer || finalAnswer;
            }
          }
        }
      }

      if (done) {
        break;
      }
    }

    if (buffer.trim()) {
      const event = parseSseBlock(buffer);
      const payload = event ? parseEventPayload(event) : null;
      if (payload) {
        sawFinal = handleStreamEvent(payload, assistantBubble) || sawFinal;
        if (payload.session_id) {
          state.activeSessionId = payload.session_id;
        }
        if (payload.type === "final") {
          finalAnswer = payload.answer || finalAnswer;
        }
      }
    }

    if (!sawFinal) {
      assistantBubble.innerHTML = renderMarkdown(finalAnswer || "流式响应已结束，但没有收到 final 事件。");
    }
  }

  function parseSseBlock(block) {
    const lines = block.split(/\r?\n/);
    let eventName = "message";
    const data = [];

    lines.forEach((line) => {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        data.push(line.slice(5).trimStart());
      }
    });

    if (!data.length) {
      return null;
    }

    return { event: eventName, data: data.join("\n") };
  }

  function parseEventPayload(event) {
    try {
      return JSON.parse(event.data);
    } catch (error) {
      return { type: event.event, answer: event.data };
    }
  }

  function handleStreamEvent(payload, assistantBubble) {
    if (payload.runtime) {
      renderRuntime(payload.runtime);
    }
    if (typeof payload.elapsed === "number") {
      els.elapsedText.textContent = formatElapsed(payload.elapsed);
    }

    if (payload.type === "error") {
      const message = payload.detail || "运行时发生错误";
      assistantBubble.innerHTML = renderMarkdown(`运行失败：${message}`);
      throw new Error(message);
    }

    if (payload.type === "message_delta") {
      const baseText = payload.reset ? "" : assistantBubble.dataset.streamText || "";
      const nextText = payload.answer || `${baseText}${payload.delta || ""}`;
      assistantBubble.dataset.streamText = nextText;
      assistantBubble.innerHTML = renderMarkdown(nextText);
      scrollMessagesToBottom();
      return false;
    }

    if (payload.type === "node_update" && payload.answer) {
      assistantBubble.dataset.streamText = payload.answer;
      assistantBubble.innerHTML = renderMarkdown(payload.answer);
      scrollMessagesToBottom();
      return false;
    }

    if (payload.type === "final") {
      assistantBubble.dataset.streamText = payload.answer || "";
      assistantBubble.innerHTML = renderMarkdown(payload.answer || "没有返回可展示内容。");
      scrollMessagesToBottom();
      return true;
    }

    return false;
  }

  function getSelectedModel() {
    return els.modelSelect.value || "glm-4.5-air";
  }

  function appendMessage(role, content, pending = false) {
    const empty = els.messageList.querySelector(".empty-state");
    if (empty) {
      empty.remove();
    }

    const message = document.createElement("article");
    message.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "你" : "行";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = pending
      ? '<span class="typing" aria-label="正在输入"><span></span><span></span><span></span></span>'
      : renderMarkdown(content);

    message.append(avatar, bubble);
    els.messageList.appendChild(message);
    scrollMessagesToBottom();
    return bubble;
  }

  function renderMessages(messages) {
    els.messageList.innerHTML = "";
    if (!messages.length) {
      els.messageList.innerHTML = `
        <div class="empty-state">
          <h2>你想怎么出发？</h2>
          <p>说目的地、日期、天数、同行人和偏好，我会帮你整理成清楚的路线。</p>
        </div>
      `;
      return;
    }

    messages.forEach((message) => {
      appendMessage(message.role === "user" ? "user" : "assistant", message.content || "");
    });
  }

  function renderRuntime(runtime) {
    els.runtimeList.innerHTML = "";

    STEPS.forEach((step) => {
      const item = runtime?.[step.key] || {
        title: step.title,
        status: "pending",
        status_label: "待开始",
        duration: 0,
        note: "-",
      };

      const status = item.status || "pending";
      const displayStatus = STATUS_TEXT[status] || item.status_label || status;
      const displayNote = formatRuntimeNote(step.key, item.note || "-", status);
      const row = document.createElement("div");
      row.className = `runtime-item ${status}`;
      row.innerHTML = `
        <div class="runtime-dot"></div>
        <div>
          <h3>${escapeHtml(step.title)}</h3>
          <div class="runtime-meta">
            <span>${escapeHtml(displayStatus)}</span>
            <span>${formatDuration(item.duration)}</span>
          </div>
          <div class="runtime-note">${escapeHtml(displayNote)}</div>
        </div>
      `;
      els.runtimeList.appendChild(row);
    });
  }

  async function refreshKnowledgeStatus() {
    try {
      const payload = await fetchJson("/api/knowledge-base/status");
      updateKnowledgeStatus(payload);
    } catch (error) {
      els.kbBadge.textContent = "连接失败";
      els.kbBadge.classList.remove("is-loaded");
      els.kbStatusText.textContent = error.message;
    }
  }

  function updateKnowledgeStatus(payload) {
    const loaded = Boolean(payload.loaded);
    const count = payload.chunk_count || 0;
    els.kbBadge.textContent = loaded ? "已加载" : "未加载";
    els.kbBadge.classList.toggle("is-loaded", loaded);
    els.kbStatusText.textContent = loaded
      ? `已收纳 ${count} 条资料片段，规划时会一起参考。`
      : "上传攻略、笔记或表格，规划时会一起参考。";
  }

  async function uploadKnowledgeFiles(event) {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (!files.length) {
      return;
    }

    const formData = new FormData();
    formData.append("model", getSelectedModel());
    files.forEach((file) => formData.append("files", file));

    els.uploadKbBtn.disabled = true;
    els.uploadKbBtn.textContent = "上传中...";
    try {
      const payload = await fetchJson("/api/knowledge-base/files", {
        method: "POST",
        body: formData,
      });
      updateKnowledgeStatus(payload);
      showToast(payload.message || "知识库已更新");
      await refreshHealth();
    } catch (error) {
      showToast(`知识库上传失败：${error.message}`);
    } finally {
      els.uploadKbBtn.disabled = false;
      els.uploadKbBtn.textContent = "上传文档";
    }
  }

  async function clearKnowledgeBase() {
    if (!window.confirm("确定清空知识库吗？")) {
      return;
    }

    try {
      const payload = await fetchJson("/api/knowledge-base", { method: "DELETE" });
      updateKnowledgeStatus(payload);
      showToast(payload.message || "知识库已清空");
      await refreshHealth();
    } catch (error) {
      showToast(`清空知识库失败：${error.message}`);
    }
  }

  function setBusy(isBusy) {
    state.isBusy = isBusy;
    els.sendBtn.disabled = isBusy;
    els.attachBtn.disabled = isBusy;
    els.newSessionBtn.disabled = isBusy;
    els.clearSessionBtn.disabled = isBusy || !state.activeSessionId;
    els.sendBtn.textContent = isBusy ? "整理中..." : "发送";
  }

  function formatRuntimeNote(stepKey, note, status) {
    if (status === "pending") {
      return "等待前一步完成";
    }
    if (status === "running") {
      return "正在处理，请稍等";
    }
    if (note.includes("由路由策略跳过")) {
      return "这次需求不需要此步骤";
    }
    if (stepKey === "router") {
      const intent = note.match(/intent=([^,\s]+)/)?.[1];
      return INTENT_TEXT[intent] || "已理解你的需求";
    }
    if (stepKey === "researcher") {
      if (note.includes("已直接答复")) {
        return "已找到可直接回答的信息";
      }
      return "已整理可参考的旅行资料";
    }
    if (stepKey === "planner") {
      return "路线内容已生成";
    }
    if (stepKey === "ticket_agent") {
      return note === "-" ? "已查询车票信息" : note;
    }
    return note === "-" ? "已完成" : note;
  }

  function resizeTextarea(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }

  function scrollMessagesToBottom() {
    requestAnimationFrame(() => {
      els.messageList.scrollTop = els.messageList.scrollHeight;
    });
  }

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.classList.add("show");
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => {
      els.toast.classList.remove("show");
    }, 3000);
  }

  function renderMarkdown(source) {
    const text = String(source || "");
    if (!text.trim()) {
      return "";
    }

    const segments = text.split(/(```[\s\S]*?```)/g);
    return segments
      .map((segment) => {
        if (segment.startsWith("```")) {
          return renderCodeBlock(segment);
        }
        return renderTextBlock(segment);
      })
      .join("");
  }

  function renderCodeBlock(segment) {
    const match = segment.match(/^```[^\n]*\n?([\s\S]*?)```$/);
    const code = match ? match[1] : segment.replace(/^```|```$/g, "");
    return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
  }

  function renderTextBlock(text) {
    const lines = text.replace(/\r\n/g, "\n").split("\n");
    const html = [];
    let paragraph = [];
    let list = null;

    const flushParagraph = () => {
      if (!paragraph.length) {
        return;
      }
      html.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
      paragraph = [];
    };

    const flushList = () => {
      if (!list) {
        return;
      }
      html.push(`<${list.type}>${list.items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${list.type}>`);
      list = null;
    };

    lines.forEach((line) => {
      const trimmed = line.trim();

      if (!trimmed) {
        flushParagraph();
        flushList();
        return;
      }

      const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        flushParagraph();
        flushList();
        const level = heading[1].length;
        html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        return;
      }

      const unordered = trimmed.match(/^[-*]\s+(.+)$/);
      const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
      if (unordered || ordered) {
        flushParagraph();
        const type = ordered ? "ol" : "ul";
        if (!list || list.type !== type) {
          flushList();
          list = { type, items: [] };
        }
        list.items.push((unordered || ordered)[1]);
        return;
      }

      flushList();
      paragraph.push(trimmed);
    });

    flushParagraph();
    flushList();
    return html.join("");
  }

  function renderInline(text) {
    let html = escapeHtml(text);
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noreferrer">$1</a>',
    );
    return html;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatDate(value) {
    if (!value) {
      return "未知时间";
    }
    return String(value).replace("T", " ").slice(0, 16);
  }

  function formatDuration(value) {
    const number = Number(value || 0);
    return number > 0 ? `${number.toFixed(2)}s` : "0s";
  }

  function formatElapsed(value) {
    const number = Number(value || 0);
    return number > 0 ? `耗时 ${number.toFixed(2)}s` : "等待任务";
  }
})();
