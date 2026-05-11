# UI.py
import streamlit as st
import inspect

def apply_custom_styles() -> None:
    """注入页面样式，提升视觉层次与移动端可读性。"""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=ZCOOL+XiaoWei&display=swap');

:root {
    --bg-top: #fff8ef;
    --bg-mid: #f3fbff;
    --bg-bottom: #fff3ea;
    --card-bg: rgba(255, 255, 255, 0.78);
    --card-border: rgba(225, 145, 80, 0.24);
    --title-text: #0f172a;
    --body-text: #334155;
    --accent: #0f766e;
    --accent-soft: #14b8a6;
    --shadow: 0 14px 35px rgba(15, 23, 42, 0.1);
}

html,
body,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 380px at -10% -10%, #ffe8cc 0%, transparent 55%),
        radial-gradient(900px 330px at 100% 0%, #c8f0ff 0%, transparent 60%),
        linear-gradient(165deg, var(--bg-top) 0%, var(--bg-mid) 52%, var(--bg-bottom) 100%);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.9) 0%, rgba(246, 251, 255, 0.95) 100%);
    border-right: 1px solid rgba(15, 118, 110, 0.18);
}

[data-testid="stSidebar"] > div:first-child {
    backdrop-filter: blur(8px);
}

h1,
h2,
h3,
.stTitle {
    font-family: "ZCOOL XiaoWei", "Noto Sans SC", sans-serif !important;
    color: var(--title-text);
}

p,
label,
[data-testid="stMarkdownContainer"],
.stCaption,
.stChatMessage {
    font-family: "Noto Sans SC", sans-serif !important;
    color: var(--body-text);
}

[data-testid="stChatMessage"] {
    border-radius: 18px;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    box-shadow: var(--shadow);
    padding: 0.35rem 0.8rem;
    margin-bottom: 0.75rem;
}

[data-testid="stChatInput"] {
    border-radius: 14px;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(20, 184, 166, 0.42);
    background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 100%);
    color: #0f172a;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent);
    box-shadow: 0 8px 18px rgba(20, 184, 166, 0.25);
    transform: translateY(-1px);
}

[data-baseweb="tag"] {
    border-radius: 999px !important;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.25rem 0.6rem;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )

def build_unified_chat_input():
    """构建单一聊天输入框，并尽量开启图片/语音能力。"""
    placeholder = "问我任何旅游问题，可直接附图或语音"
    try:
        params = inspect.signature(st.chat_input).parameters
    except Exception:
        return st.chat_input(placeholder), False

    kwargs = {}
    supports_unified_upload = False
    if "accept_file" in params:
        kwargs["accept_file"] = "multiple"
        supports_unified_upload = True
        if "file_type" in params:
            # kwargs["file_type"] = ["image", "audio"]
           
            # 明确指定支持的图片和音频扩展名
            kwargs["file_type"] = ["png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "ogg"]

    if "accept_audio" in params:
        kwargs["accept_audio"] = True
        supports_unified_upload = True
        if "audio_sample_rate" in params:
            kwargs["audio_sample_rate"] = 16000

    try:
        return st.chat_input(placeholder, **kwargs), supports_unified_upload
    except TypeError:
        return st.chat_input(placeholder), False




def render_sidebar(config):
    """渲染侧边栏（模型选择、会话管理、知识库），返回选中的模型与上传文件"""
    with st.sidebar:
        st.header("设置面板")
        try:
            default_idx = config.MODEL_LIST.index(st.session_state.current_model)
        except ValueError:
            default_idx = 0

        selected_model = st.selectbox("对话模型", config.MODEL_LIST, index=default_idx)
        st.divider()
        st.subheader("会话管理")
        if st.button("新建会话", use_container_width=True):
            st.session_state.session_counter += 1
            new_name = f"新会话{st.session_state.session_counter}"
            st.session_state.sessions[new_name] = []
            st.session_state.current_session = new_name
            st.session_state.needs_save = True
            st.rerun()

        session_names = list(st.session_state.sessions.keys())
        selected_session = st.selectbox(
            "历史会话",
            session_names,
            index=session_names.index(st.session_state.current_session),
            label_visibility="collapsed",
        )
        col1, col2 = st.columns(2)
        with col1:
            clear_clicked = st.button("清空")
        with col2:
            delete_clicked = st.button("删除")

        st.divider()
        st.subheader("知识库")
        if st.session_state.vector_db is not None:
            chunk_count = st.session_state.vector_db._collection.count()
            st.success(f"知识库已加载，共 {chunk_count} 个文本块。")
        else:
            st.info("未加载知识库，Agent 将只使用内置工具和模型能力。")

        if st.session_state.kb_notice:
            kb_notice = st.session_state.kb_notice
            notice_text = f"[{kb_notice.get('at', '')}] {kb_notice.get('message', '')}"
            notice_level = kb_notice.get("level", "info")
            if notice_level == "error":
                st.error(notice_text)
            elif notice_level == "warning":
                st.warning(notice_text)
            else:
                st.success(notice_text)
            if st.button("清除本次提示", use_container_width=True, key="clear_kb_notice"):
                st.session_state.kb_notice = None
                st.rerun()

        uploaded_files = st.file_uploader(
            "上传文档扩充知识库（攻略、游记等）",
            type=["pdf", "txt", "docx", "csv"],
            accept_multiple_files=True,
        )
        ingest_clicked = st.button("开始入库", use_container_width=True)
        clear_db_clicked = st.button("清空知识库", use_container_width=True)

        return (
            selected_model, selected_session,
            clear_clicked, delete_clicked,
            uploaded_files, ingest_clicked, clear_db_clicked
        )

def render_chat_history(messages):
    """渲染聊天历史"""
    for msg in messages:
        st.chat_message(msg["role"]).write(msg["content"])

def render_page_title():
    """渲染页面标题"""
    st.set_page_config(page_title="旅游规划助手", layout="wide")
    apply_custom_styles()
    st.title("🗺️ 旅游规划助手")
    st.caption("支持天气查询、景点搜索、路线规划、行程生成，可上传攻略文档增强回答")