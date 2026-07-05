(function () {
  "use strict";

  const ACCESS_TOKEN_KEY = "travelAssistant.accessToken";
  const REFRESH_TOKEN_KEY = "travelAssistant.refreshToken";
  const AUTH_USER_KEY = "travelAssistant.authUser";
  const RUNTIME_WIDTH_KEY = "travelAssistant.runtimeWidth";
  const SESSION_CACHE_PREFIX = "travelAssistant.sessionCache.";
  const ACTIVE_SESSION_KEY = "travelAssistant.activeSessionId";
  const TRIP_BUILDER_OPEN_KEY = "travelAssistant.tripBuilderOpen";
  const DEFAULT_API_BASE = "http://127.0.0.1:8000";
  const API_BASE_CANDIDATES = buildApiBaseCandidates();
  const MULTI_TRIP_FIELDS = new Set(["travelers", "preferences", "localMobility"]);
  const QUICK_PROMPT_PAGE_SIZE = 3;
  const XHS_IMPORT_COOLDOWN_MS = 60 * 1000;

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

  const QUICK_PROMPTS = [
    {
      title: "周末慢游",
      meta: "2 天 · 休闲 · 少走路",
      prompt: "我想周末去无锡玩 2 天，偏好休闲、美食和少走路，请帮我安排一条节奏舒服的路线。",
    },
    {
      title: "亲子轻松线",
      meta: "3 天 · 亲子 · 安全便利",
      prompt: "我想带孩子去无锡玩 3 天，偏好亲子、自然和美食，希望路线安全、转场少、餐饮方便。",
    },
    {
      title: "老人友好线",
      meta: "3 天 · 老人 · 无障碍优先",
      prompt: "我想带老人去无锡玩 3 天，预算舒适，要求少走路、休息点多、无障碍优先，请帮我规划。",
    },
    {
      title: "摄影夜游线",
      meta: "2 天 · 摄影 · 夜游",
      prompt: "我想去无锡玩 2 天，喜欢摄影、夜游和小众街区，请安排适合拍照且晚上不无聊的路线。",
    },
    {
      title: "省钱学生线",
      meta: "2 天 · 经济 · 公共交通",
      prompt: "我想去无锡玩 2 天，预算经济，偏好美食和人文，尽量用公共交通和免费或低价景点。",
    },
    {
      title: "品质度假线",
      meta: "4 天 · 品质 · 慢节奏",
      prompt: "我想去无锡玩 4 天，预算品质，偏好自然、休闲和好吃的餐厅，希望少排队、体验更舒服。",
    },
    {
      title: "临时改行程",
      meta: "追问 · 调整 · 继续规划",
      prompt: "刚才的行程我想调整一下：把天数改成 4 天，减少赶路，多安排一些适合休息和吃饭的地方。",
    },
    {
      title: "车票查询",
      meta: "查询 · 到达方式 · 时间衔接",
      prompt: "帮我查询明天从南京到无锡的高铁票，并结合到达时间建议第一天怎么安排比较顺。",
    },
    {
      title: "资料库推荐",
      meta: "知识库 · 对比 · 总结",
      prompt: "请根据我上传到知识库的资料，推荐适合亲子和老人一起出行的目的地，并说明理由。",
    },
  ];

  function buildApiBaseCandidates() {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    const hostname = window.location.hostname || "127.0.0.1";
    const configured = window.TRAVEL_ASSIST_API_BASE || "";
    return [...new Set([
      configured,
      `${protocol}//${hostname}:8000`,
      `${protocol}//${hostname}:8001`,
      DEFAULT_API_BASE,
    ].filter(Boolean).map(normalizeApiBase))];
  }

  const state = {
    apiBase: DEFAULT_API_BASE,
    activeSessionId: null,
    activeSession: null,
    sessions: [],
    selectedFiles: [],
    tripDraft: createEmptyTripDraft(),
    tripDraftDirty: false,
    latestItineraryMarkdown: "",
    latestRouteRequest: "",
    latestRuntime: null,
    latestElapsedText: "",
    quickPromptsOpen: false,
    quickPromptCursor: 0,
    xhsCooldownUntil: 0,
    accessToken: "",
    refreshToken: "",
    authUser: null,
    authMode: "login",
    isRefreshingToken: false,
    isBusy: false,
    shouldAutoScroll: true,
    toastTimer: null,
  };

  const els = {
    healthStatus: document.getElementById("healthStatus"),
    sessionList: document.getElementById("sessionList"),
    newSessionBtn: document.getElementById("newSessionBtn"),
    activeSessionTitle: document.getElementById("activeSessionTitle"),
    activeSessionMeta: document.getElementById("activeSessionMeta"),
    modelSelect: document.getElementById("modelSelect"),
    accountName: document.getElementById("accountName"),
    accountPopover: document.getElementById("accountPopover"),
    logoutBtn: document.getElementById("logoutBtn"),
    authModal: document.getElementById("authModal"),
    authForm: document.getElementById("authForm"),
    authTitle: document.getElementById("authTitle"),
    authSubtitle: document.getElementById("authSubtitle"),
    authUsername: document.getElementById("authUsername"),
    authPassword: document.getElementById("authPassword"),
    authSubmitBtn: document.getElementById("authSubmitBtn"),
    authToggleModeBtn: document.getElementById("authToggleModeBtn"),
    authStatusText: document.getElementById("authStatusText"),
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
    quickPrompts: document.getElementById("quickPrompts"),
    quickPromptList: document.getElementById("quickPromptList"),
    shufflePromptBtn: document.getElementById("shufflePromptBtn"),
    elapsedText: document.getElementById("elapsedText"),
    clearSessionBtn: document.getElementById("clearSessionBtn"),
    kbBadge: document.getElementById("kbBadge"),
    kbStatusText: document.getElementById("kbStatusText"),
    kbHintText: document.getElementById("kbHintText"),
    kbNoticeText: document.getElementById("kbNoticeText"),
    kbFileInput: document.getElementById("kbFileInput"),
    kbSourceTabs: Array.from(document.querySelectorAll("[data-kb-source]")),
    kbSourcePanels: Array.from(document.querySelectorAll("[data-kb-panel]")),
    xhsUrlInput: document.getElementById("xhsUrlInput"),
    uploadKbBtn: document.getElementById("uploadKbBtn"),
    importXhsBtn: document.getElementById("importXhsBtn"),
    clearKbBtnGlobal: document.getElementById("clearKbBtnGlobal"),
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
    state.apiBase = DEFAULT_API_BASE;
    restoreAuthState();

    bindEvents();
    restoreRuntimeWidth();
    restoreTripBuilderState();
    renderTripSummary();
    renderTripSummaryPanel();
    renderQuickPrompts();
    renderRuntime();
    renderSelectedFiles();
    renderAuthState();
    setBusy(false);

    await detectApiBase();
    await Promise.allSettled([loadModels(), refreshHealth()]);
    if (state.accessToken || state.refreshToken) {
      await verifyAuthSession();
    } else {
      showAuthModal("login");
      renderMessages([]);
      updateSessionHeader();
    }
  }

  function bindEvents() {
    els.newSessionBtn.addEventListener("click", createSession);
    els.authForm.addEventListener("submit", handleAuthSubmit);
    els.authToggleModeBtn.addEventListener("click", () => {
      showAuthModal(state.authMode === "login" ? "register" : "login");
    });
    els.logoutBtn.addEventListener("click", logout);
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
    els.messageInput.addEventListener("focus", () => setQuickPromptsOpen(true));

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
    els.importXhsBtn.addEventListener("click", importXhsUrl);
    els.clearKbBtnGlobal.addEventListener("click", clearKnowledgeBase);
    els.kbSourceTabs.forEach((button) => {
      button.addEventListener("click", () => setKnowledgeSource(button.getAttribute("data-kb-source") || "file"));
    });
    els.xhsUrlInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        importXhsUrl();
      }
    });

    els.quickPromptList.addEventListener("click", (event) => {
      const button = event.target.closest("[data-prompt]");
      if (!button) {
        return;
      }
      fillPrompt(button.getAttribute("data-prompt") || "");
    });
    els.shufflePromptBtn.addEventListener("click", (event) => {
      event.preventDefault();
      shuffleQuickPrompts();
      setQuickPromptsOpen(true);
    });

    els.sidebarToggleBtn.addEventListener("click", () => {
      els.sidebar.classList.toggle("open");
    });

    initColumnResizer();
    initRuntimeTabs();

    document.addEventListener("click", (event) => {
      const clickedPromptArea = els.quickPrompts?.contains(event.target) || els.messageInput.contains(event.target);
      if (!clickedPromptArea) {
        setQuickPromptsOpen(false);
      }

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

  function setKnowledgeSource(source) {
    setKnowledgeNotice("");
    els.kbSourceTabs.forEach((button) => {
      const isActive = button.getAttribute("data-kb-source") === source;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", String(isActive));
    });
    els.kbSourcePanels.forEach((panel) => {
      panel.classList.toggle("active", panel.getAttribute("data-kb-panel") === source);
    });
  }

  function fillPrompt(prompt) {
    els.messageInput.value = prompt;
    resizeTextarea(els.messageInput);
    els.messageInput.focus();
    setQuickPromptsOpen(false);
  }

  function setQuickPromptsOpen(open) {
    state.quickPromptsOpen = Boolean(open);
    if (!els.quickPrompts) {
      return;
    }
    els.quickPrompts.hidden = !state.quickPromptsOpen;
    els.quickPrompts.classList.toggle("open", state.quickPromptsOpen);
  }

  function renderQuickPrompts() {
    if (!els.quickPromptList) {
      return;
    }
    const prompts = getVisibleQuickPrompts();
    els.quickPromptList.innerHTML = prompts.map((item) => `
      <button class="quick-prompt-card" type="button" data-prompt="${escapeHtml(item.prompt)}">
        <span>${escapeHtml(item.meta)}</span>
        <strong>${escapeHtml(item.title)}</strong>
        <small>${escapeHtml(item.prompt)}</small>
      </button>
    `).join("");
  }

  function getVisibleQuickPrompts() {
    const visible = [];
    for (let offset = 0; offset < QUICK_PROMPT_PAGE_SIZE; offset += 1) {
      visible.push(QUICK_PROMPTS[(state.quickPromptCursor + offset) % QUICK_PROMPTS.length]);
    }
    return visible;
  }

  function shuffleQuickPrompts() {
    state.quickPromptCursor = (state.quickPromptCursor + QUICK_PROMPT_PAGE_SIZE) % QUICK_PROMPTS.length;
    renderQuickPrompts();
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
    if (draft.departure) {
      parts.push(`${draft.departure}出发`);
    }
    if (draft.days) {
      parts.push(draft.days);
    }
    if (draft.startDate) {
      parts.push(draft.startDate);
    }
    if (draft.arrivalMode) {
      parts.push(draft.arrivalMode);
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
    els.tripDraftSummary.innerHTML = parts.length
      ? parts.map((part) => `<span>${escapeHtml(part)}</span>`).join("")
      : "未选择标签，可直接聊天";
  }

  function renderTripSummaryPanel() {
    if (!els.tripSummaryPanel) {
      return;
    }

    const draftItems = getTripDraftItems();
    const latestRouteRequest = getLatestRouteRequest();
    const latestTitle = getItineraryTitle(state.latestItineraryMarkdown);
    const hasDraft = draftItems.length > 0;
    const canRegenerate = Boolean(latestTitle);

    const draftHtml = hasDraft
      ? draftItems.map((item) => `
          <div class="summary-row">
            <span>${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(item.value)}</strong>
          </div>
        `).join("")
      : '<p class="summary-empty">还没有选择标签。展开左侧“快速描述需求”，可以先把目的地、天数、预算和交通偏好搭起来。</p>';

    const requestHtml = latestRouteRequest
      ? `<div class="summary-request">${renderMarkdown(latestRouteRequest)}</div>`
      : '<p class="summary-empty">发送需求后，这里会保留最近一次规划依据。</p>';

    const itineraryHtml = latestTitle
      ? `
          <div class="summary-latest">
          <span>最新行程</span>
          <strong>${escapeHtml(latestTitle)}</strong>
          <div class="summary-actions">
            <button type="button" data-summary-action="copy">复制</button>
            <button type="button" data-summary-action="export">导出 MD</button>
            <button type="button" data-summary-action="regenerate">重新生成</button>
          </div>
        </div>
      `
      : `
        <div class="summary-latest">
          <span>结果操作</span>
          <strong>${canRegenerate ? "可基于当前上下文再生成一版" : "生成行程后，可在这里快速复制或导出 Markdown。"}</strong>
          ${canRegenerate ? `
            <div class="summary-actions">
              <button type="button" data-summary-action="regenerate">重新生成</button>
            </div>
          ` : ""}
        </div>
      `;

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
    els.tripSummaryPanel.querySelector("[data-summary-action='regenerate']")?.addEventListener("click", () => {
      regenerateItinerary().catch((error) => {
        showToast(`重新生成失败：${error.message}`);
      });
    });
  }

  function getTripDraftItems() {
    const draft = state.tripDraft;
    return [
      { label: "目的地", value: draft.destination },
      { label: "天数", value: draft.days },
      { label: "出发日期", value: draft.startDate },
      { label: "出发地", value: draft.departure },
      { label: "到达方式", value: draft.arrivalMode },
      { label: "同行", value: draft.travelers.join("、") },
      { label: "偏好", value: draft.preferences.join("、") },
      { label: "预算", value: draft.budget },
      { label: "当地交通", value: draft.localMobility.join("、") },
    ].filter((item) => isMeaningfulSummaryValue(item.value));
  }

  function getLatestRouteRequest() {
    if (state.latestRouteRequest) {
      return state.latestRouteRequest;
    }
    const messages = state.activeSession?.messages || [];
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const content = messages[index]?.content || "";
      if (messages[index]?.role === "user" && isLikelyPlanningRequest(content)) {
        return summarizeRequestText(content);
      }
    }
    return "";
  }

  function isLikelyPlanningRequest(text) {
    const compact = String(text || "").replace(/\s+/g, "");
    return /旅行|旅游|出行|行程|路线|攻略|我想去|目的地|玩\d{1,2}天|偏好|预算|同行人/.test(compact);
  }

  function isRegenerateRequest(text) {
    return /^请基于本会话已经确认的出行需求和上下文，重新生成一版不同的行程方案。/.test(String(text || "").trim());
  }

  function summarizeRequestText(text) {
    const lines = String(text || "")
      .split(/\n+/)
      .map(cleanSummaryLine)
      .filter(Boolean);
    return lines.slice(0, 6).join("\n");
  }

  function buildRouteRequestSummary(update = {}) {
    const intent = String(update.intent || "");
    const departure = cleanSummaryValue(update.departure);
    const city = cleanSummaryValue(update.city);
    const companions = cleanSummaryValue(update.companions);
    const days = Number(update.days || 0);
    const startDate = cleanSummaryValue(update.start_date);
    const preference = cleanSummaryValue(update.preference);
    const userQuery = String(update.user_query || "").trim();

    if (!["need_plan", "need_more_info"].includes(intent)) {
      return state.latestRouteRequest || summarizeRequestText(userQuery);
    }

    const parts = [];
    if (departure && city) {
      parts.push(`${departure}出发 -> ${city}${days > 0 ? ` ${days}日` : ""}`);
    } else {
      parts.push(city ? `${city}${days > 0 ? ` ${days}日` : ""}` : days > 0 ? `${days}日行程` : "旅行规划");
    }
    if (companions) {
      parts.push(companions);
    }
    if (startDate === "日期灵活") {
      parts.push("日期不定");
    } else if (startDate) {
      parts.push(`${startDate}出发`);
    }
    if (preference) {
      parts.push(preference.replace(/\+/g, " / "));
    }

    const summary = parts.filter(Boolean).join(" · ");
    if (summary && summary !== "旅行规划") {
      return summary;
    }
    return summarizeRequestText(userQuery);
  }

  function formatMissingField(field) {
    return {
      departure: "出发地",
      city: "目的地",
      companions: "同行人",
      days: "天数",
      start_date: "出发日期",
      preference: "偏好",
    }[field] || field;
  }

  function cleanSummaryLine(line) {
    const text = String(line || "").trim();
    if (!text) {
      return "";
    }
    const cleanedParts = text
      .replace(/[。.]$/, "")
      .split("；")
      .map((part) => part.trim())
      .filter((part) => !/[：:]\s*(未指定|空字符|空字符串|无|没有|无偏好|不限)\s*$/.test(part));
    return cleanedParts.length ? `${cleanedParts.join("；")}。` : "";
  }

  function cleanSummaryValue(value) {
    const text = String(value || "").trim();
    return isMeaningfulSummaryValue(text) ? text : "";
  }

  function isMeaningfulSummaryValue(value) {
    const text = String(value || "").trim();
    return Boolean(text) && !["未指定", "空字符", "空字符串", "无", "没有", "无偏好", "不限"].includes(text);
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

    const transferParts = [];
    if (draft.departure) {
      transferParts.push(`出发地：${draft.departure}`);
    }
    if (draft.arrivalMode) {
      transferParts.push(`到达方式：${draft.arrivalMode}`);
    }
    if (transferParts.length) {
      lines.push(`${transferParts.join("；")}。`);
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

  async function detectApiBase() {
    setHealth("checking", "正在连接服务");
    const fallbackCandidates = [];

    for (const candidate of API_BASE_CANDIDATES) {
      try {
        const response = await fetch(`${candidate}/api/auth/me`, {
          headers: { Accept: "application/json" },
        });
        if (response.status === 401) {
          state.apiBase = candidate;
          return candidate;
        }
        fallbackCandidates.push(candidate);
      } catch (error) {
        // Try the next local candidate.
      }
    }

    for (const candidate of fallbackCandidates.length ? fallbackCandidates : API_BASE_CANDIDATES) {
      try {
        const response = await fetch(`${candidate}/api/health`, {
          headers: { Accept: "application/json" },
        });
        if (response.ok) {
          state.apiBase = candidate;
          return candidate;
        }
      } catch (error) {
        // Try the next local candidate.
      }
    }

    state.apiBase = DEFAULT_API_BASE;
    setHealth("error", "未找到后端服务");
    return state.apiBase;
  }

  function restoreAuthState() {
    state.accessToken = localStorage.getItem(ACCESS_TOKEN_KEY) || "";
    state.refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY) || "";
    try {
      state.authUser = JSON.parse(localStorage.getItem(AUTH_USER_KEY) || "null");
    } catch (error) {
      state.authUser = null;
    }
  }

  function saveAuthState(payload) {
    state.accessToken = payload.access_token || "";
    state.refreshToken = payload.refresh_token || state.refreshToken || "";
    state.authUser = payload.user || state.authUser;
    if (state.accessToken) {
      localStorage.setItem(ACCESS_TOKEN_KEY, state.accessToken);
    }
    if (state.refreshToken) {
      localStorage.setItem(REFRESH_TOKEN_KEY, state.refreshToken);
    }
    if (state.authUser) {
      localStorage.setItem(AUTH_USER_KEY, JSON.stringify(state.authUser));
    }
    renderAuthState();
  }

  function clearAuthState() {
    state.accessToken = "";
    state.refreshToken = "";
    state.authUser = null;
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
    localStorage.removeItem(ACTIVE_SESSION_KEY);
    state.activeSessionId = null;
    state.activeSession = null;
    state.sessions = [];
    renderAuthState();
    renderSessions();
    renderMessages([]);
    updateSessionHeader();
  }

  function renderAuthState() {
    const username = state.authUser?.username || "";
    els.accountName.textContent = username ? `账号：${username}` : "未登录";
    els.logoutBtn.hidden = !username;
    renderAccountPopover();
  }

  function renderAccountPopover() {
    if (!els.accountPopover) {
      return;
    }

    if (!state.authUser) {
      els.accountPopover.innerHTML = `
        <strong>未登录</strong>
        <p>登录后可以保存旅行记录和资料库状态。</p>
      `;
      return;
    }

    const stats = getPlanningStats();
    els.accountPopover.innerHTML = `
      <div class="account-popover-head">
        <strong>${escapeHtml(state.authUser.username || "旅行用户")}</strong>
        <span>${escapeHtml(stats.lastPlanText)}</span>
      </div>
      <div class="account-metrics">
        <div><strong>${stats.totalPlans}</strong><span>旅行记录</span></div>
        <div><strong>${stats.activeDays}</strong><span>活跃日期</span></div>
      </div>
      <div class="account-calendar">
        <div class="account-calendar-title">${escapeHtml(stats.monthTitle)}</div>
        <div class="account-calendar-grid">
          ${stats.weekdays.map((day) => `<span class="calendar-weekday">${day}</span>`).join("")}
          ${stats.cells.map((cell) => `
            <span class="${cell.className}" title="${escapeHtml(cell.title)}">${cell.label}</span>
          `).join("")}
        </div>
      </div>
    `;
  }

  function getPlanningStats() {
    const sessions = Array.isArray(state.sessions) ? state.sessions : [];
    const plannedSessions = sessions.filter((session) => Number(session.message_count || 0) > 0);
    const dates = plannedSessions
      .map((session) => parseLocalDate(session.updated_at))
      .filter(Boolean)
      .sort((a, b) => b.getTime() - a.getTime());
    const lastDate = dates[0] || null;
    const lastPlanText = lastDate ? formatDaysSince(lastDate) : "还没有开始规划";
    const calendar = buildActivityCalendar(dates);

    return {
      totalPlans: plannedSessions.length,
      activeDays: new Set(dates.map(dateKey)).size,
      lastPlanText,
      ...calendar,
    };
  }

  function buildActivityCalendar(dates) {
    const today = new Date();
    const year = today.getFullYear();
    const month = today.getMonth();
    const activeKeys = new Set(
      dates
        .filter((date) => date.getFullYear() === year && date.getMonth() === month)
        .map(dateKey),
    );
    const firstDay = new Date(year, month, 1);
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const offset = (firstDay.getDay() + 6) % 7;
    const cells = [];

    for (let index = 0; index < offset; index += 1) {
      cells.push({ label: "", className: "calendar-day is-empty", title: "" });
    }

    for (let day = 1; day <= daysInMonth; day += 1) {
      const current = new Date(year, month, day);
      const key = dateKey(current);
      const isActive = activeKeys.has(key);
      const isToday = key === dateKey(today);
      cells.push({
        label: String(day),
        className: [
          "calendar-day",
          isActive ? "is-active" : "",
          isToday ? "is-today" : "",
        ].filter(Boolean).join(" "),
        title: isActive ? `${key} 有旅行记录` : key,
      });
    }

    return {
      monthTitle: `${year} 年 ${month + 1} 月`,
      weekdays: ["一", "二", "三", "四", "五", "六", "日"],
      cells,
    };
  }

  function showAuthModal(mode = state.authMode || "login", message = "") {
    state.authMode = mode;
    const isRegister = mode === "register";
    els.authTitle.textContent = isRegister ? "注册账号" : "登录账号";
    els.authSubtitle.textContent = isRegister ? "创建账号后，会自动建立你的旅行记录空间。" : "登录后保存你的旅行记录和规划资料。";
    els.authSubmitBtn.textContent = isRegister ? "注册并登录" : "登录";
    els.authToggleModeBtn.textContent = isRegister ? "已有账号？去登录" : "没有账号？注册一个";
    setAuthStatus(message, message ? "error" : "");
    els.authModal.classList.add("open");
    window.setTimeout(() => els.authUsername.focus(), 0);
  }

  function hideAuthModal() {
    els.authModal.classList.remove("open");
    setAuthStatus("");
  }

  function setAuthStatus(message, type = "") {
    const text = String(message || "").trim();
    els.authStatusText.hidden = !text;
    els.authStatusText.textContent = text;
    els.authStatusText.classList.toggle("is-error", type === "error");
    els.authStatusText.classList.toggle("is-success", type === "success");
  }

  function isPublicAuthPath(path) {
    return ["/api/auth/login", "/api/auth/register", "/api/auth/refresh"].includes(String(path || ""));
  }

  function withAuthHeaders(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const isForm = options.body instanceof FormData;
    if (!isForm && options.body !== undefined && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    if (state.accessToken && !headers.has("Authorization") && !isPublicAuthPath(path)) {
      headers.set("Authorization", `Bearer ${state.accessToken}`);
    }
    return { ...options, headers };
  }

  async function fetchWithAuth(path, options = {}, retry = true) {
    const response = await fetch(apiUrl(path), withAuthHeaders(path, options));
    if (response.status === 401 && retry && !isPublicAuthPath(path) && state.refreshToken) {
      const refreshed = await refreshAccessToken();
      if (refreshed) {
        return fetchWithAuth(path, options, false);
      }
    }
    return response;
  }

  async function fetchJson(path, options = {}) {
    const response = await fetchWithAuth(path, options);
    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }
    return response.json();
  }

  async function refreshAccessToken() {
    if (!state.refreshToken || state.isRefreshingToken) {
      return false;
    }
    state.isRefreshingToken = true;
    try {
      const response = await fetch(apiUrl("/api/auth/refresh"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: state.refreshToken }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }
      saveAuthState(await response.json());
      return true;
    } catch (error) {
      clearAuthState();
      showAuthModal("login", error.message || "登录状态已过期，请重新登录。");
      return false;
    } finally {
      state.isRefreshingToken = false;
    }
  }

  async function verifyAuthSession() {
    try {
      const payload = await fetchJson("/api/auth/me");
      state.authUser = payload.user || state.authUser;
      if (state.authUser) {
        localStorage.setItem(AUTH_USER_KEY, JSON.stringify(state.authUser));
      }
      renderAuthState();
      hideAuthModal();
      await Promise.allSettled([loadSessions(), refreshKnowledgeStatus()]);
    } catch (error) {
      clearAuthState();
      showAuthModal("login", error.message);
    }
  }

  async function handleAuthSubmit(event) {
    event.preventDefault();
    const username = els.authUsername.value.trim();
    const password = els.authPassword.value;
    const authValidationError = validateAuthFields(username, password);
    if (authValidationError) {
      setAuthStatus(authValidationError, "error");
      return;
    }

    els.authSubmitBtn.disabled = true;
    els.authSubmitBtn.textContent = state.authMode === "register" ? "注册中..." : "登录中...";
    setAuthStatus("");

    try {
      const payload = await fetchJson(`/api/auth/${state.authMode}`, {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      saveAuthState(payload);
      els.authPassword.value = "";
      hideAuthModal();
      setAuthStatus("登录成功。", "success");
      await Promise.allSettled([loadSessions(null, { resetRuntime: true }), refreshKnowledgeStatus()]);
      showToast(state.authMode === "register" ? "注册成功，已登录" : "登录成功");
    } catch (error) {
      setAuthStatus(error.message, "error");
    } finally {
      els.authSubmitBtn.disabled = false;
      els.authSubmitBtn.textContent = state.authMode === "register" ? "注册并登录" : "登录";
    }
  }

  function validateAuthFields(username, password) {
    if (!/^[A-Za-z0-9_]{3,32}$/.test(username)) {
      return "用户名需为 3-32 位，仅支持字母、数字和下划线。";
    }
    if (!/^(\S){8,64}$/.test(password)) {
      return "密码长度需要在 8 到 64 位之间，且不能包含空格。";
    }
    if (!/[A-Za-z]/.test(password) || !/\d/.test(password)) {
      return "密码需要同时包含字母和数字。";
    }
    return "";
  }

  async function logout() {
    const refreshToken = state.refreshToken;
    try {
      if (refreshToken) {
        await fetchJson("/api/auth/logout", {
          method: "POST",
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
      }
    } catch (error) {
      // 本地退出优先，服务端撤销失败时也不保留本地登录态。
    } finally {
      clearAuthState();
      setKnowledgeNotice("");
      showAuthModal("login");
      showToast("已退出登录");
    }
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
      void payload;
      setHealth("ok", "已连接");
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
      renderAccountPopover();

      if (nextSessionId) {
        await loadSession(nextSessionId, options);
      } else {
        state.activeSessionId = null;
        state.activeSession = null;
        state.latestRouteRequest = "";
        state.latestItineraryMarkdown = "";
        state.latestRuntime = null;
        state.latestElapsedText = "";
        renderMessages([]);
        renderRuntime();
        updateSessionHeader();
      }
    } catch (error) {
      showToast(`会话加载失败：${error.message}`);
      renderSessions();
      renderAccountPopover();
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
    renderAccountPopover();
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
        <div class="session-title">${escapeHtml(sanitizeTitle(session.title) || "新会话")}</div>
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
      restoreSessionCache();
      renderSessions();
      renderAccountPopover();
      renderMessages(payload.session.messages || []);
      updateSessionHeader();
      if (options.resetRuntime !== false) {
        renderRuntime(state.latestRuntime || undefined);
        els.elapsedText.textContent = state.latestElapsedText || "还没有开始";
      }
      setBusy(state.isBusy);
    } catch (error) {
      showToast(`会话读取失败：${error.message}`);
    }
  }

  function getSessionCacheKey(sessionId = state.activeSessionId) {
    return sessionId ? `${SESSION_CACHE_PREFIX}${sessionId}` : "";
  }

  function persistSessionCache() {
    const key = getSessionCacheKey();
    if (!key) {
      return;
    }
    const payload = {
      latestRouteRequest: state.latestRouteRequest || "",
      latestItineraryMarkdown: state.latestItineraryMarkdown || "",
      latestRuntime: state.latestRuntime || null,
      latestElapsedText: state.latestElapsedText || "",
    };
    localStorage.setItem(key, JSON.stringify(payload));
  }

  function restoreSessionCache() {
    state.latestRouteRequest = "";
    state.latestItineraryMarkdown = "";
    state.latestRuntime = null;
    state.latestElapsedText = "";

    const key = getSessionCacheKey();
    if (!key) {
      return;
    }
    try {
      const cached = JSON.parse(localStorage.getItem(key) || "null");
      if (!cached || typeof cached !== "object") {
        return;
      }
      state.latestRouteRequest = String(cached.latestRouteRequest || "");
      state.latestItineraryMarkdown = String(cached.latestItineraryMarkdown || "");
      state.latestRuntime = cached.latestRuntime && typeof cached.latestRuntime === "object" ? cached.latestRuntime : null;
      state.latestElapsedText = String(cached.latestElapsedText || "");
    } catch (error) {
      localStorage.removeItem(key);
    }
  }

  function clearSessionCache(sessionId = state.activeSessionId) {
    const key = getSessionCacheKey(sessionId);
    if (key) {
      localStorage.removeItem(key);
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
      clearSessionCache(sessionId);
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
      state.latestRouteRequest = "";
      state.latestItineraryMarkdown = "";
      state.latestRuntime = null;
      state.latestElapsedText = "";
      clearSessionCache(state.activeSessionId);
      renderMessages([]);
      renderRuntime();
      els.elapsedText.textContent = "还没有开始";
      updateSessionHeader();
      await loadSessions(state.activeSessionId);
      showToast("当前会话已清空");
    } catch (error) {
      showToast(`清空会话失败：${error.message}`);
    }
  }

  function updateSessionHeader() {
    const session = state.activeSession;
    els.activeSessionTitle.textContent = sanitizeTitle(session?.title) || "新会话";
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

    await submitPreparedMessage(message, files);
  }

  async function submitPreparedMessage(message, files = []) {
    setQuickPromptsOpen(false);
    clearGuidanceActions();
    els.messageInput.value = "";
    resizeTextarea(els.messageInput);
    state.tripDraftDirty = false;
    state.selectedFiles = [];
    renderSelectedFiles();

    const displayText = files.length ? `${message || "已上传文件"}\n\n${files.map((file) => `- ${file.name}`).join("\n")}` : message;
    if (isLikelyPlanningRequest(message) && !isRegenerateRequest(message)) {
      state.latestRouteRequest = summarizeRequestText(message);
      persistSessionCache();
    }
    appendMessage("user", displayText);
    rememberMessage("user", displayText);
    renderTripSummaryPanel();
    const assistantBubble = appendMessage("assistant", "", true);
    state.shouldAutoScroll = true;
    scrollMessagesToBottom(true);
    setBusy(true);
    renderRuntime();
    state.latestRuntime = null;
    state.latestElapsedText = "";
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

  async function regenerateItinerary() {
    if (state.isBusy) {
      return;
    }
    if (!state.activeSessionId && !state.latestItineraryMarkdown) {
      showToast("先生成一次行程，再重新生成");
      return;
    }

    const supplement = els.messageInput.value.trim();
    const displayText = supplement
      ? `按补充要求重新生成一版行程。\n补充要求：${supplement}`
      : "重新生成一版不同的行程方案。";

    els.messageInput.value = "";
    resizeTextarea(els.messageInput);
    setQuickPromptsOpen(false);
    appendMessage("user", displayText);
    rememberMessage("user", displayText);
    renderTripSummaryPanel();

    const assistantBubble = appendMessage("assistant", "", true);
    state.shouldAutoScroll = true;
    scrollMessagesToBottom(true);
    setBusy(true);
    renderRuntime();
    els.elapsedText.textContent = "正在重新生成";

    try {
      await sendRegenerateStreamingChat(supplement, assistantBubble);
      await Promise.allSettled([refreshSessionSummaries(), refreshHealth(), refreshKnowledgeStatus()]);
    } catch (error) {
      setBubbleContent(assistantBubble, "assistant", `重新生成失败：${error.message}`);
      showToast(`重新生成失败：${error.message}`);
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
    state.latestRuntime = payload.runtime || state.latestRuntime;
    state.latestElapsedText = formatElapsed(payload.elapsed);
    renderRuntime(payload.runtime);
    els.elapsedText.textContent = formatElapsed(payload.elapsed);
    applyRouteUpdatesFromEvents(payload.events || []);
    persistSessionCache();
  }

  async function sendStreamingChat(message, assistantBubble) {
    return sendStreamingRequest("/api/chat/stream", {
      message,
      session_id: state.activeSessionId,
      model: getSelectedModel(),
      save_to_session: true,
    }, assistantBubble);
  }

  async function sendRegenerateStreamingChat(supplement, assistantBubble) {
    return sendStreamingRequest("/api/chat/regenerate/stream", {
      session_id: state.activeSessionId,
      model: getSelectedModel(),
      supplement: supplement || "",
    }, assistantBubble);
  }

  async function sendStreamingRequest(endpoint, body, assistantBubble) {
    const response = await fetchWithAuth(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
            if (payload.session_id) {
              state.activeSessionId = payload.session_id;
              localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
              if (state.activeSession) {
                state.activeSession.id = state.activeSessionId;
              }
            }
            sawFinal = handleStreamEvent(payload, assistantBubble) || sawFinal;
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
        if (payload.session_id) {
          state.activeSessionId = payload.session_id;
          localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
          if (state.activeSession) {
            state.activeSession.id = state.activeSessionId;
          }
        }
        sawFinal = handleStreamEvent(payload, assistantBubble) || sawFinal;
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
      state.latestRuntime = payload.runtime;
      renderRuntime(payload.runtime);
      persistSessionCache();
    }
    if (typeof payload.elapsed === "number") {
      state.latestElapsedText = formatElapsed(payload.elapsed);
      els.elapsedText.textContent = state.latestElapsedText;
      persistSessionCache();
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

    if (payload.type === "node_update" && payload.node === "router" && payload.state_update) {
      applyRouteStateUpdate(payload.state_update);
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
      persistSessionCache();
      scrollMessagesToBottom();
      return true;
    }

    return false;
  }

  function applyRouteUpdatesFromEvents(events) {
    if (!Array.isArray(events)) {
      return;
    }
    events.forEach((event) => {
      if (event?.type === "node_update" && event.node === "router" && event.state_update) {
        applyRouteStateUpdate(event.state_update);
      }
    });
  }

  function applyRouteStateUpdate(update) {
    const summary = buildRouteRequestSummary(update);
    if (!summary) {
      renderTripSummary();
      renderTripSummaryPanel();
      return;
    }
    state.latestRouteRequest = summary;
    renderTripSummary();
    renderTripSummaryPanel();
    persistSessionCache();
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
    const displayText = role === "assistant" && !pending ? stripGuidanceActionLine(text) : text;
    bubble.innerHTML = pending
      ? '<span class="typing" aria-label="正在输入"><span></span><span></span><span></span></span>'
      : renderMarkdown(displayText);
    renderGuidanceActions(bubble, role, text, actions && !pending);
    renderResultActions(bubble, role, text, actions && !pending);
  }

  function stripGuidanceActionLine(content) {
    return String(content || "")
      .split(/\n/)
      .filter((line) => !/^\s*可选[：:]\s*(?:\[[^\]]+\]\s*)+$/.test(line))
      .join("\n")
      .trim();
  }

  function renderGuidanceActions(bubble, role, content, shouldRender) {
    const body = bubble.parentElement;
    if (!body) {
      return;
    }
    body.querySelector(".guidance-actions")?.remove();
    if (!shouldRender || role !== "assistant") {
      return;
    }
    const actions = extractGuidanceActions(content);
    if (!actions.length) {
      return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "guidance-actions";
    actions.forEach((action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = action.label;
      button.addEventListener("click", () => submitGuidanceAction(action.value));
      wrapper.appendChild(button);
    });
    body.appendChild(wrapper);
  }

  function clearGuidanceActions() {
    els.messageList.querySelectorAll(".guidance-actions").forEach((actions) => actions.remove());
  }

  function extractGuidanceActions(content) {
    const text = String(content || "");
    const matches = [...text.matchAll(/\[([^\]]{1,12})\]/g)]
      .map((match) => match[1].trim())
      .filter(Boolean);
    const unique = [...new Set(matches)];
    return unique.map((label) => {
      if (label.startsWith("选") && label.length > 1) {
        const city = label.slice(1);
        return { label, value: `我选择${city}作为目的地。` };
      }
      if (label === "换一换") {
        return { label, value: "换一换目的地推荐。" };
      }
      return { label, value: label };
    });
  }

  function submitGuidanceAction(value) {
    if (state.isBusy) {
      return;
    }
    submitPreparedMessage(value, []).catch((error) => {
      showToast(`发送失败：${error.message}`);
    });
  }

  function renderResultActions(bubble, role, content, shouldRender) {
    const body = bubble.parentElement;
    if (!body) {
      return;
    }
    body.querySelector(".result-actions")?.remove();
    if (!shouldRender || role !== "assistant") {
      return;
    }

    const itineraryContent = isItineraryContent(content);
    if (!itineraryContent) {
      return;
    }
    state.latestItineraryMarkdown = content;
    renderTripSummaryPanel();
    persistSessionCache();

    const actions = document.createElement("div");
    actions.className = "result-actions";

    if (itineraryContent) {
      const copyButton = document.createElement("button");
      copyButton.type = "button";
      copyButton.textContent = "复制";
      copyButton.addEventListener("click", () => copyMarkdown(content));

      const exportButton = document.createElement("button");
      exportButton.type = "button";
      exportButton.textContent = "导出 MD";
      exportButton.addEventListener("click", () => exportMarkdown(content));

      actions.append(copyButton, exportButton);
    }

    const regenerateButton = document.createElement("button");
    regenerateButton.type = "button";
    regenerateButton.textContent = "重新生成";
    regenerateButton.title = "基于当前会话上下文重新生成一版行程；输入框中的补充要求会一并带上";
    regenerateButton.addEventListener("click", () => {
      regenerateItinerary().catch((error) => {
        showToast(`重新生成失败：${error.message}`);
      });
    });

    actions.append(regenerateButton);
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
    return sanitizeTitle(text.match(/^#\s+(.+)$/m)?.[1]?.trim() || "");
  }

  function sanitizeTitle(title) {
    return String(title || "")
      .replace(/\s*·\s*(?:空字符|空字符串|未指定|无偏好|无|没有|不限)\s*$/g, "")
      .replace(/(?:空字符|空字符串)/g, "")
      .replace(/\s{2,}/g, " ")
      .trim();
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
      els.kbHintText.textContent = "请先确认后端服务已启动。";
      setKnowledgeNotice(error.message, "error");
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
    els.kbHintText.textContent = loaded ? "继续导入会合并到同一个知识库。" : "支持 PDF、DOCX、TXT、CSV 和小红书笔记链接。";
  }

  function setKnowledgeNotice(message, type = "info") {
    if (!els.kbNoticeText) {
      return;
    }
    const text = String(message || "").trim();
    els.kbNoticeText.hidden = !text;
    els.kbNoticeText.textContent = text;
    els.kbNoticeText.classList.toggle("is-error", type === "error");
    els.kbNoticeText.classList.toggle("is-success", type === "success");
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
    setKnowledgeNotice("");
    try {
      const payload = await fetchJson("/api/knowledge-base/files", {
        method: "POST",
        body: formData,
      });
      updateKnowledgeStatus(payload);
      setKnowledgeNotice(payload.message || "知识库已更新", "success");
      showToast(payload.message || "知识库已更新");
      await refreshHealth();
    } catch (error) {
      setKnowledgeNotice(error.message, "error");
      showToast(`知识库上传失败：${error.message}`);
    } finally {
      els.uploadKbBtn.disabled = false;
      els.uploadKbBtn.textContent = "选择文件";
    }
  }

  async function importXhsUrl() {
    const url = (els.xhsUrlInput.value || "").trim();
    if (!url) {
      showToast("先粘贴一条小红书笔记链接");
      els.xhsUrlInput.focus();
      return;
    }

    const waitMs = state.xhsCooldownUntil - Date.now();
    if (waitMs > 0) {
      const waitSeconds = Math.ceil(waitMs / 1000);
      setKnowledgeNotice(`小红书导入已进入保护间隔，请 ${waitSeconds} 秒后再试。`, "error");
      showToast(`小红书导入保护中：${waitSeconds} 秒后再试`);
      return;
    }

    els.importXhsBtn.disabled = true;
    els.importXhsBtn.textContent = "导入中...";
    state.xhsCooldownUntil = Date.now() + XHS_IMPORT_COOLDOWN_MS;
    setKnowledgeNotice("");
    try {
      const payload = await fetchJson("/api/knowledge-base/xhs-url", {
        method: "POST",
        body: JSON.stringify({
          url,
          model: getSelectedModel(),
        }),
      });
      updateKnowledgeStatus(payload);
      els.xhsUrlInput.value = "";
      setKnowledgeNotice(payload.message || "小红书笔记已导入", "success");
      showToast(payload.message || "小红书笔记已导入");
      await refreshHealth();
    } catch (error) {
      setKnowledgeNotice(error.message, "error");
      showToast(`小红书导入失败：${error.message}`);
    } finally {
      els.importXhsBtn.disabled = false;
      els.importXhsBtn.textContent = "导入链接";
    }
  }

  async function clearKnowledgeBase() {
    if (!window.confirm("确定清空知识库吗？")) {
      return;
    }

    try {
      const payload = await fetchJson("/api/knowledge-base", { method: "DELETE" });
      updateKnowledgeStatus(payload);
      setKnowledgeNotice(payload.message || "知识库已清空", "success");
      showToast(payload.message || "知识库已清空");
      await refreshHealth();
    } catch (error) {
      setKnowledgeNotice(error.message, "error");
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

  function parseLocalDate(value) {
    if (!value) {
      return null;
    }
    const text = String(value).trim().replace("T", " ");
    const match = text.match(/^(\d{4})-(\d{2})-(\d{2})(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?/);
    if (!match) {
      return null;
    }
    const [, year, month, day, hour = "0", minute = "0", second = "0"] = match;
    const date = new Date(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second),
    );
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function dateKey(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  function formatDaysSince(date) {
    const today = new Date();
    const startOfToday = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    const startOfDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    const days = Math.max(0, Math.floor((startOfToday - startOfDate) / 86400000));
    if (days === 0) {
      return "今天刚计划过旅行";
    }
    if (days === 1) {
      return "你已经 1 天没有计划旅行了";
    }
    return `你已经 ${days} 天没有计划旅行了`;
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
