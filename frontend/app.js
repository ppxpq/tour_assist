(function () {
  "use strict";

  const API_BASE_KEY = "travelAssistant.apiBase";
  const RUNTIME_WIDTH_KEY = "travelAssistant.runtimeWidth";
  const ACTIVE_SESSION_KEY = "travelAssistant.activeSessionId";
  const TRIP_BUILDER_OPEN_KEY = "travelAssistant.tripBuilderOpen";
  const DEFAULT_API_BASE = "http://127.0.0.1:8000";
  const MULTI_TRIP_FIELDS = new Set(["travelers", "preferences", "localMobility"]);

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
    tripDraft: createEmptyTripDraft(),
    tripDraftDirty: false,
    latestItineraryMarkdown: "",
    isBusy: false,
    shouldAutoScroll: true,
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
    tripDestinationInput: document.getElementById("tripDestinationInput"),
    tripStartDateInput: document.getElementById("tripStartDateInput"),
    tripDepartureInput: document.getElementById("tripDepartureInput"),
    tripDaysInput: document.getElementById("tripDaysInput"),
    tripBuilderToggleBtn: document.getElementById("tripBuilderToggleBtn"),
    tripBuilderBody: document.getElementById("tripBuilderBody"),
    tripDraftSummary: document.getElementById("tripDraftSummary"),
    clearTripDraftBtn: document.getElementById("clearTripDraftBtn"),
    tripButtons: Array.from(document.querySelectorAll("[data-trip-field]")),
    sendBtn: document.getElementById("sendBtn"),
    attachBtn: document.getElementById("attachBtn"),
    chatFileInput: document.getElementById("chatFileInput"),
    selectedFiles: document.getElementById("selectedFiles"),
    runtimeList: document.getElementById("runtimeList"),
    runtimeTabs: Array.from(document.querySelectorAll("[data-runtime-tab]")),
    runtimePanels: Array.from(document.querySelectorAll("[data-runtime-panel]")),
    tripSummaryPanel: document.getElementById("tripSummaryPanel"),
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
    workspace: document.querySelector(".workspace"),
    columnResizer: document.getElementById("columnResizer"),
  };

  document.addEventListener("DOMContentLoaded", init);

  function createEmptyTripDraft() {
    return {
      destination: "",
      startDate: "",
      departure: "",
      days: "",
      travelers: [],
      preferences: [],
      budget: "",
      arrivalMode: "",
      localMobility: [],
    };
  }

  async function init() {
    state.apiBase = normalizeApiBase(localStorage.getItem(API_BASE_KEY) || DEFAULT_API_BASE);
    els.apiBaseInput.value = state.apiBase;

    bindEvents();
    restoreRuntimeWidth();
    restoreTripBuilderState();
    renderTripSummary();
    renderTripSummaryPanel();
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
    els.tripBuilderToggleBtn.addEventListener("click", toggleTripBuilder);
    els.clearTripDraftBtn.addEventListener("click", clearTripDraft);
    els.tripDestinationInput.addEventListener("input", () => {
      state.tripDraft.destination = els.tripDestinationInput.value.trim();
      state.tripDraftDirty = true;
      renderTripSummary();
      renderTripSummaryPanel();
    });
    els.tripStartDateInput.addEventListener("input", () => {
      state.tripDraft.startDate = els.tripStartDateInput.value;
      state.tripDraftDirty = true;
      renderTripSummary();
      renderTripSummaryPanel();
    });
    els.tripDepartureInput.addEventListener("input", () => {
      state.tripDraft.departure = els.tripDepartureInput.value.trim();
      state.tripDraftDirty = true;
      renderTripSummary();
      renderTripSummaryPanel();
    });
    els.tripDaysInput.addEventListener("input", () => {
      const days = Number(els.tripDaysInput.value);
      const normalizedDays = Number.isFinite(days) && days >= 5 ? Math.min(days, 30) : 0;
      if (normalizedDays && String(normalizedDays) !== els.tripDaysInput.value) {
        els.tripDaysInput.value = String(normalizedDays);
      }
      state.tripDraft.days = normalizedDays ? `${normalizedDays}天` : "";
      state.tripDraftDirty = true;
      renderTripTags();
      renderTripSummary();
      renderTripSummaryPanel();
    });
    els.tripButtons.forEach((button) => {
      button.addEventListener("click", () => toggleTripTag(button));
    });

    els.messageInput.addEventListener("input", () => {
      resizeTextarea(els.messageInput);
    });

    els.messageList.addEventListener("scroll", () => {
      const distanceFromBottom = els.messageList.scrollHeight - els.messageList.scrollTop - els.messageList.clientHeight;
      state.shouldAutoScroll = distanceFromBottom < 80;
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

    initColumnResizer();
    initRuntimeTabs();

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

  function toggleTripTag(button) {
    const field = button.getAttribute("data-trip-field");
    const value = button.getAttribute("data-trip-value") || "";
    if (!field || !value) {
      return;
    }

    if (MULTI_TRIP_FIELDS.has(field)) {
      const current = state.tripDraft[field] || [];
      state.tripDraft[field] = current.includes(value)
        ? current.filter((item) => item !== value)
        : [...current, value];
    } else {
      state.tripDraft[field] = state.tripDraft[field] === value ? "" : value;
      if (field === "days") {
        els.tripDaysInput.value = "";
      }
    }

    state.tripDraftDirty = true;
    renderTripTags();
    renderTripSummary();
    renderTripSummaryPanel();
  }

  function renderTripTags() {
    els.tripButtons.forEach((button) => {
      const field = button.getAttribute("data-trip-field");
      const value = button.getAttribute("data-trip-value") || "";
      const current = state.tripDraft[field];
      const isActive = Array.isArray(current) ? current.includes(value) : current === value;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });
  }

  function toggleTripBuilder() {
    const isOpen = els.tripBuilderToggleBtn.getAttribute("aria-expanded") === "true";
    setTripBuilderOpen(!isOpen);
  }

  function setTripBuilderOpen(isOpen) {
    els.tripBuilderToggleBtn.setAttribute("aria-expanded", String(isOpen));
    els.tripBuilderBody.hidden = !isOpen;
    localStorage.setItem(TRIP_BUILDER_OPEN_KEY, isOpen ? "true" : "false");
    const caret = els.tripBuilderToggleBtn.querySelector(".toggle-caret");
    if (caret) {
      caret.textContent = isOpen ? "▾" : "▸";
    }
  }

  function restoreTripBuilderState() {
    const stored = localStorage.getItem(TRIP_BUILDER_OPEN_KEY);
    setTripBuilderOpen(stored === "true");
  }

  function renderTripSummary() {
    const draft = state.tripDraft;
    const parts = [];
    if (draft.destination) {
      parts.push(draft.destination);
    }
    if (draft.days) {
      parts.push(draft.days);
    }
    if (draft.startDate) {
      parts.push(draft.startDate);
    }
    if (draft.travelers.length) {
      parts.push(draft.travelers.join("/"));
    }
    if (draft.preferences.length) {
      parts.push(draft.preferences.slice(0, 3).join("/"));
    }
    if (draft.budget) {
      parts.push(`${draft.budget}预算`);
    }
    if (draft.localMobility.length) {
      parts.push(draft.localMobility.slice(0, 2).join("/"));
    }
    els.tripDraftSummary.textContent = parts.length ? parts.join(" · ") : "未选择标签，可直接聊天";
  }

  function renderTripSummaryPanel() {
    if (!els.tripSummaryPanel) {
      return;
    }

    const draftItems = getTripDraftItems();
    const latestUserMessage = getLatestUserMessage();
    const latestTitle = getItineraryTitle(state.latestItineraryMarkdown);
    const hasDraft = draftItems.some((item) => item.value !== "未指定");

    const draftHtml = hasDraft
      ? draftItems.map((item) => `
          <div class="summary-row">
            <span>${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(item.value)}</strong>
          </div>
        `).join("")
      : '<p class="summary-empty">还没有选择标签。展开左侧“快速描述需求”，可以先把目的地、天数、预算和交通偏好搭起来。</p>';

    const requestHtml = latestUserMessage
      ? `<div class="summary-request">${renderMarkdown(latestUserMessage)}</div>`
      : '<p class="summary-empty">发送需求后，这里会保留最近一次规划依据。</p>';

    const itineraryHtml = latestTitle
      ? `
        <div class="summary-latest">
          <span>最新行程</span>
          <strong>${escapeHtml(latestTitle)}</strong>
          <div class="summary-actions">
            <button type="button" data-summary-action="copy">复制</button>
            <button type="button" data-summary-action="export">导出 MD</button>
          </div>
        </div>
      `
      : '<p class="summary-empty">生成行程后，可在这里快速复制或导出 Markdown。</p>';

    els.tripSummaryPanel.innerHTML = `
      <section class="summary-card">
        <h3>当前标签</h3>
        ${draftHtml}
      </section>
      <section class="summary-card">
        <h3>最新路线</h3>
        ${requestHtml}
      </section>
      <section class="summary-card">
        <h3>结果操作</h3>
        ${itineraryHtml}
      </section>
    `;

    els.tripSummaryPanel.querySelector("[data-summary-action='copy']")?.addEventListener("click", () => {
      copyMarkdown(state.latestItineraryMarkdown);
    });
    els.tripSummaryPanel.querySelector("[data-summary-action='export']")?.addEventListener("click", () => {
      exportMarkdown(state.latestItineraryMarkdown);
    });
  }

  function getTripDraftItems() {
    const draft = state.tripDraft;
    return [
      { label: "目的地", value: draft.destination || "未指定" },
      { label: "天数", value: draft.days || "未指定" },
      { label: "出发日期", value: draft.startDate || "未指定" },
      { label: "出发地", value: draft.departure || "未指定" },
      { label: "到达方式", value: draft.arrivalMode || "未指定" },
      { label: "同行", value: draft.travelers.join("、") || "未指定" },
      { label: "偏好", value: draft.preferences.join("、") || "未指定" },
      { label: "预算", value: draft.budget || "未指定" },
      { label: "当地交通", value: draft.localMobility.join("、") || "未指定" },
    ];
  }

  function getLatestUserMessage() {
    const messages = state.activeSession?.messages || [];
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === "user" && messages[index]?.content) {
        return messages[index].content;
      }
    }
    return "";
  }

  function clearTripDraft() {
    state.tripDraft = createEmptyTripDraft();
    els.tripDestinationInput.value = "";
    els.tripStartDateInput.value = "";
    els.tripDepartureInput.value = "";
    els.tripDaysInput.value = "";
    state.tripDraftDirty = false;
    renderTripTags();
    renderTripSummary();
    renderTripSummaryPanel();
    showToast("已清空标签选择");
  }

  function hasTripDraft() {
    const draft = state.tripDraft;
    return Boolean(
      draft.destination ||
      draft.startDate ||
      draft.departure ||
      draft.days ||
      draft.budget ||
      draft.arrivalMode ||
      draft.travelers.length ||
      draft.preferences.length ||
      draft.localMobility.length,
    );
  }

  function buildTripPrompt(freeText, useTripDraft = true) {
    const draft = state.tripDraft;
    const supplement = (freeText || "").trim();
    if (!useTripDraft || !hasTripDraft()) {
      return supplement;
    }

    const lines = [];
    const firstLine = [
      draft.destination ? `我想去${draft.destination}` : "我想规划一次出行",
      draft.days ? `玩${draft.days}` : "",
      draft.startDate ? `${draft.startDate}出发` : "",
    ].filter(Boolean).join("，");
    lines.push(`${firstLine}。`);

    if (draft.departure || draft.arrivalMode) {
      lines.push(`出发地：${draft.departure || "未指定"}；到达方式：${draft.arrivalMode || "未指定"}。`);
    }
    if (draft.travelers.length) {
      lines.push(`同行人：${draft.travelers.join("、")}。`);
    }
    if (draft.preferences.length) {
      lines.push(`偏好：${draft.preferences.join("、")}。`);
    }
    if (draft.budget) {
      lines.push(`预算：${draft.budget}。`);
    }
    if (draft.localMobility.length) {
      lines.push(`当地交通偏好：${draft.localMobility.join("、")}。`);
    }
    if (supplement) {
      lines.push(`补充要求：${supplement}`);
    }
    return lines.join("\n");
  }

  function restoreRuntimeWidth() {
    const stored = Number(localStorage.getItem(RUNTIME_WIDTH_KEY) || 0);
    if (Number.isFinite(stored) && stored >= 260) {
      setRuntimeWidth(stored);
    }
  }

  function setRuntimeWidth(width) {
    const workspaceWidth = els.workspace?.getBoundingClientRect().width || 0;
    const maxWidth = workspaceWidth ? Math.max(260, Math.min(560, workspaceWidth - 520)) : 560;
    const nextWidth = Math.round(Math.max(260, Math.min(maxWidth, width)));
    document.documentElement.style.setProperty("--runtime-width", `${nextWidth}px`);
    return nextWidth;
  }

  function initColumnResizer() {
    if (!els.columnResizer || !els.workspace) {
      return;
    }

    let startX = 0;
    let startWidth = 0;

    const stopResize = () => {
      document.body.classList.remove("column-resizing");
      els.workspace.classList.remove("resizing");
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", stopResize);
      window.removeEventListener("pointercancel", stopResize);
    };

    const onPointerMove = (event) => {
      const nextWidth = setRuntimeWidth(startWidth - (event.clientX - startX));
      localStorage.setItem(RUNTIME_WIDTH_KEY, String(nextWidth));
    };

    els.columnResizer.addEventListener("pointerdown", (event) => {
      if (window.matchMedia("(max-width: 1100px)").matches) {
        return;
      }
      event.preventDefault();
      startX = event.clientX;
      startWidth = document.querySelector(".runtime-column")?.getBoundingClientRect().width || 340;
      document.body.classList.add("column-resizing");
      els.workspace.classList.add("resizing");
      els.columnResizer.setPointerCapture?.(event.pointerId);
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", stopResize);
      window.addEventListener("pointercancel", stopResize);
    });

    els.columnResizer.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        return;
      }
      event.preventDefault();
      const currentWidth = document.querySelector(".runtime-column")?.getBoundingClientRect().width || 340;
      let nextWidth = currentWidth;
      if (event.key === "ArrowLeft") {
        nextWidth = currentWidth + 24;
      } else if (event.key === "ArrowRight") {
        nextWidth = currentWidth - 24;
      } else if (event.key === "Home") {
        nextWidth = 260;
      } else if (event.key === "End") {
        nextWidth = 520;
      }
      localStorage.setItem(RUNTIME_WIDTH_KEY, String(setRuntimeWidth(nextWidth)));
    });
  }

  function initRuntimeTabs() {
    els.runtimeTabs.forEach((button) => {
      button.addEventListener("click", () => {
        setRuntimeTab(button.getAttribute("data-runtime-tab") || "progress");
      });
    });
  }

  function setRuntimeTab(tabName) {
    els.runtimeTabs.forEach((button) => {
      const isActive = button.getAttribute("data-runtime-tab") === tabName;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", String(isActive));
    });
    els.runtimePanels.forEach((panel) => {
      const isActive = panel.getAttribute("data-runtime-panel") === tabName;
      panel.classList.toggle("active", isActive);
    });
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

  async function loadSessions(preferredSessionId = state.activeSessionId || localStorage.getItem(ACTIVE_SESSION_KEY), options = {}) {
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
      localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
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
      if (state.activeSessionId) {
        localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
      }
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
      if (state.activeSessionId) {
        localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
      } else {
        localStorage.removeItem(ACTIVE_SESSION_KEY);
      }
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

    const rawMessage = els.messageInput.value.trim();
    const shouldUseTripDraft = state.tripDraftDirty || (!rawMessage && hasTripDraft());
    const message = buildTripPrompt(rawMessage, shouldUseTripDraft);
    const files = [...state.selectedFiles];
    if (!message && !files.length) {
      showToast("先说一点旅行想法，或上传图片/语音");
      return;
    }

    els.messageInput.value = "";
    resizeTextarea(els.messageInput);
    state.tripDraftDirty = false;
    state.selectedFiles = [];
    renderSelectedFiles();

    const displayText = files.length ? `${message || "已上传文件"}\n\n${files.map((file) => `- ${file.name}`).join("\n")}` : message;
    appendMessage("user", displayText);
    rememberMessage("user", displayText);
    renderTripSummaryPanel();
    const assistantBubble = appendMessage("assistant", "", true);
    state.shouldAutoScroll = true;
    scrollMessagesToBottom(true);
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
      setBubbleContent(assistantBubble, "assistant", `请求失败：${error.message}`);
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
    if (state.activeSessionId) {
      localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
      if (state.activeSession) {
        state.activeSession.id = state.activeSessionId;
      }
    }
    setBubbleContent(assistantBubble, "assistant", payload.message || "没有返回可展示内容。");
    rememberMessage("assistant", payload.message || "");
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
      setBubbleContent(assistantBubble, "assistant", text || "浏览器不支持流式读取。");
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
              localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
              if (state.activeSession) {
                state.activeSession.id = state.activeSessionId;
              }
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
          localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
          if (state.activeSession) {
            state.activeSession.id = state.activeSessionId;
          }
        }
        if (payload.type === "final") {
          finalAnswer = payload.answer || finalAnswer;
        }
      }
    }

    if (!sawFinal) {
      setBubbleContent(assistantBubble, "assistant", finalAnswer || "流式响应已结束，但没有收到 final 事件。");
      rememberMessage("assistant", finalAnswer || "");
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
      setBubbleContent(assistantBubble, "assistant", `运行失败：${message}`);
      throw new Error(message);
    }

    if (payload.type === "message_delta") {
      const baseText = payload.reset ? "" : assistantBubble.dataset.streamText || "";
      const nextText = payload.answer || `${baseText}${payload.delta || ""}`;
      assistantBubble.dataset.streamText = nextText;
      setBubbleContent(assistantBubble, "assistant", nextText, { actions: false });
      scrollMessagesToBottom();
      return false;
    }

    if (payload.type === "node_update" && payload.answer) {
      assistantBubble.dataset.streamText = payload.answer;
      setBubbleContent(assistantBubble, "assistant", payload.answer, { actions: false });
      scrollMessagesToBottom();
      return false;
    }

    if (payload.type === "final") {
      assistantBubble.dataset.streamText = payload.answer || "";
      setBubbleContent(assistantBubble, "assistant", payload.answer || "没有返回可展示内容。");
      rememberMessage("assistant", payload.answer || "");
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

    const body = document.createElement("div");
    body.className = "message-body";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    body.appendChild(bubble);
    message.append(avatar, body);
    els.messageList.appendChild(message);
    setBubbleContent(bubble, role, content, { pending, actions: !pending });
    scrollMessagesToBottom();
    return bubble;
  }

  function setBubbleContent(bubble, role, content, options = {}) {
    const pending = Boolean(options.pending);
    const actions = options.actions !== false;
    const text = String(content || "");
    bubble.dataset.rawContent = pending ? "" : text;
    bubble.innerHTML = pending
      ? '<span class="typing" aria-label="正在输入"><span></span><span></span><span></span></span>'
      : renderMarkdown(text);
    renderResultActions(bubble, role, text, actions && !pending);
  }

  function renderResultActions(bubble, role, content, shouldRender) {
    const body = bubble.parentElement;
    if (!body) {
      return;
    }
    body.querySelector(".result-actions")?.remove();
    if (!shouldRender || role !== "assistant" || !isItineraryContent(content)) {
      return;
    }

    state.latestItineraryMarkdown = content;
    renderTripSummaryPanel();

    const actions = document.createElement("div");
    actions.className = "result-actions";

    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.textContent = "复制";
    copyButton.addEventListener("click", () => copyMarkdown(content));

    const exportButton = document.createElement("button");
    exportButton.type = "button";
    exportButton.textContent = "导出 MD";
    exportButton.addEventListener("click", () => exportMarkdown(content));

    actions.append(copyButton, exportButton);
    body.appendChild(actions);
  }

  function rememberMessage(role, content) {
    if (!content) {
      return;
    }
    if (!state.activeSession) {
      state.activeSession = {
        id: state.activeSessionId,
        title: "新会话",
        messages: [],
      };
    }
    state.activeSession.messages = [...(state.activeSession.messages || []), { role, content }];
  }

  function isItineraryContent(content) {
    const text = String(content || "");
    return /^#\s+.+行程/m.test(text) || /^##\s+Day\s*\d+/im.test(text) || /##\s+行程概览/.test(text);
  }

  async function copyMarkdown(content) {
    const text = String(content || "").trim();
    if (!text) {
      showToast("还没有可复制的行程");
      return;
    }
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        fallbackCopyText(text);
      }
      showToast("行程 Markdown 已复制");
    } catch (error) {
      fallbackCopyText(text);
      showToast("行程 Markdown 已复制");
    }
  }

  function fallbackCopyText(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }

  function exportMarkdown(content) {
    const text = String(content || "").trim();
    if (!text) {
      showToast("还没有可导出的行程");
      return;
    }
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${slugifyFilename(getItineraryTitle(text) || "行程规划")}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("已导出 Markdown");
  }

  function getItineraryTitle(content) {
    const text = String(content || "");
    return text.match(/^#\s+(.+)$/m)?.[1]?.trim() || "";
  }

  function slugifyFilename(value) {
    const filename = String(value || "行程规划")
      .trim()
      .replace(/[\\/:*?"<>|]/g, "-")
      .replace(/\s+/g, "-")
      .slice(0, 48);
    return filename || "行程规划";
  }

  function renderMessages(messages) {
    state.shouldAutoScroll = true;
    state.latestItineraryMarkdown = "";
    els.messageList.innerHTML = "";
    if (!messages.length) {
      els.messageList.innerHTML = `
        <div class="empty-state">
          <h2>你想怎么出发？</h2>
          <p>说目的地、日期、天数、同行人和偏好，我会帮你整理成清楚的路线。</p>
        </div>
      `;
      renderTripSummaryPanel();
      return;
    }

    messages.forEach((message) => {
      appendMessage(message.role === "user" ? "user" : "assistant", message.content || "");
    });
    renderTripSummaryPanel();
    scrollMessagesToBottom(true);
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
    els.newSessionBtn.disabled = false;
    els.clearSessionBtn.disabled = !state.activeSessionId;
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
      return "本次请求无需此步骤";
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

  function scrollMessagesToBottom(force = false) {
    if (!force && !state.shouldAutoScroll) {
      return;
    }
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
    let openSection = null;

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

    const closeSection = () => {
      if (openSection) {
        html.push("</section>");
        openSection = null;
      }
    };

    const sectionClassForHeading = (level, title) => {
      if (level === 2 && /^Day\s*\d+/i.test(title)) {
        return "itinerary-section itinerary-day";
      }
      if (level === 2 && /行程概览/.test(title)) {
        return "itinerary-section itinerary-overview";
      }
      if (level === 2 && /注意事项/.test(title)) {
        return "itinerary-section itinerary-notes";
      }
      if (level === 2 && /预算/.test(title)) {
        return "itinerary-section itinerary-budget";
      }
      if (level === 3 && /上午|下午|晚上|餐饮建议/.test(title)) {
        return "itinerary-section itinerary-slot";
      }
      return "";
    };

    lines.forEach((line) => {
      const trimmed = line.trim();

      if (!trimmed) {
        flushParagraph();
        flushList();
        return;
      }

      if (/^📅\s*日期[：:]/.test(trimmed)) {
        flushParagraph();
        flushList();
        html.push(`<p class="itinerary-meta">${renderInline(trimmed)}</p>`);
        return;
      }

      if (/^本日概要[：:]/.test(trimmed)) {
        flushParagraph();
        flushList();
        html.push(`<p class="itinerary-day-summary">${renderInline(trimmed)}</p>`);
        return;
      }

      const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        flushParagraph();
        flushList();
        const level = heading[1].length;
        const title = heading[2];
        const sectionClass = sectionClassForHeading(level, title);
        if (sectionClass) {
          closeSection();
          html.push(`<section class="${sectionClass}">`);
          openSection = sectionClass;
        } else if (level <= 2) {
          closeSection();
        }
        if (level === 1) {
          html.push(`<div class="itinerary-hero"><span>行程方案</span><h1>${renderInline(title)}</h1></div>`);
        } else {
          html.push(`<h${level}>${renderInline(title)}</h${level}>`);
        }
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
    closeSection();
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
