import gc
import hashlib
import os
import re
import shutil
import tempfile
import time
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm_core import get_embeddings
from utils import config


def _resolve_persist_path(persist_path: str | None = None) -> str:
    return persist_path or config.PERSIST_PATH


def _cleanup_persist_path(persist_path: str | None = None) -> None:
    """尽量彻底清理向量库目录，避免损坏数据残留。"""
    resolved_path = _resolve_persist_path(persist_path)
    if not os.path.exists(resolved_path):
        return

    try:
        shutil.rmtree(resolved_path)
        return
    except Exception:
        pass

    for filename in os.listdir(resolved_path):
        file_path = os.path.join(resolved_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception:
            pass


def _normalize_text_for_chunking(text: str) -> str:
    """对 OCR/PDF 常见换行和空白噪声做轻量清洗。"""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u3000", " ").replace("\xa0", " ")
    normalized = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(
        r"(?<!\n)(?<=[^\n。！？；!?;])\n(?=[^\n\-*•\d])",
        " ",
        normalized,
    )
    return normalized.strip()


def _filter_existing_chunk_ids(vector_db, chunk_map):
    """过滤已存在的 chunk id，避免重复入库。"""
    if vector_db is None or not chunk_map:
        return chunk_map, 0

    candidate_ids = list(chunk_map.keys())
    existing_ids = set()

    try:
        existing = vector_db.get(ids=candidate_ids, include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        try:
            existing = vector_db._collection.get(ids=candidate_ids, include=[])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            existing_ids = set()

    if not existing_ids:
        return chunk_map, 0

    filtered_map = {
        chunk_id: doc for chunk_id, doc in chunk_map.items() if chunk_id not in existing_ids
    }
    return filtered_map, len(existing_ids)


@lru_cache(maxsize=1)
def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """复用切分器，避免批量上传时重复初始化。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            "Day",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],
        is_separator_regex=False,
    )


def _build_loader(file_path: str, suffix: str):
    if suffix == ".pdf":
        return PyPDFLoader(file_path)
    if suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    if suffix in {".docx", ".doc"}:
        return Docx2txtLoader(file_path)
    if suffix == ".csv":
        return CSVLoader(file_path, encoding="utf-8")
    return None


def load_db(persist_path: str | None = None):
    resolved_path = _resolve_persist_path(persist_path)
    if not os.path.exists(resolved_path):
        return None

    try:
        has_data = bool(os.listdir(resolved_path))
    except Exception:
        has_data = True

    if not has_data:
        return None

    try:
        vector_db = Chroma(
            persist_directory=resolved_path,
            embedding_function=get_embeddings(),
            collection_name="my_docs",
        )
        vector_db.get(limit=1, include=[])
        return vector_db
    except Exception as exc:
        print(f"检测到向量库异常，准备重建: {exc}")
        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        _cleanup_persist_path(resolved_path)
        return None


def ingest_documents(uploaded_files, vector_db, selected_model, persist_path: str | None = None):
    """
    uploaded_files: file-like objects with .name and .getbuffer()
    vector_db: 当前 Chroma 实例，可能为 None
    selected_model: 保留参数，便于后续扩展
    """
    del selected_model
    resolved_path = _resolve_persist_path(persist_path)

    if not uploaded_files:
        return vector_db, {"success": False, "message": "没有检测到上传文件。"}

    all_chunks = []
    success_count = 0
    text_splitter = _get_text_splitter()

    for file in uploaded_files:
        temp_path = None
        suffix = os.path.splitext(file.name)[-1].lower()

        if suffix not in {".pdf", ".txt", ".docx", ".doc", ".csv"}:
            print(f"跳过不支持的文件格式: {file.name}")
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix or ".tmp", delete=False) as tmp:
                tmp.write(file.getbuffer())
                temp_path = tmp.name

            loader = _build_loader(temp_path, suffix)
            if loader is None:
                continue

            data = loader.load()
            for doc in data:
                doc.page_content = _normalize_text_for_chunking(doc.page_content)
                if not isinstance(doc.metadata, dict):
                    doc.metadata = {}
                doc.metadata["source"] = file.name

            chunks = text_splitter.split_documents(data)
            all_chunks.extend(chunks)
            success_count += 1
        except Exception as exc:
            print(f"处理文件 {file.name} 时出错: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_chunks:
        return vector_db, {"success": False, "message": "未能从上传文件中提取到有效文本。"}

    unique_chunks_dict = {}
    for chunk in all_chunks:
        content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
        if content_hash not in unique_chunks_dict:
            unique_chunks_dict[content_hash] = chunk

    final_chunks = list(unique_chunks_dict.values())
    new_chunk_map, existed_count = _filter_existing_chunk_ids(vector_db, unique_chunks_dict)
    new_chunks = list(new_chunk_map.values())
    new_ids = list(new_chunk_map.keys())

    if not new_chunks:
        return vector_db, {
            "success": True,
            "message": (
                f"成功解析 {success_count} 个文件。"
                f"共提取 {len(all_chunks)} 个文本块，批次去重后为 {len(final_chunks)} 个，"
                f"其中 {existed_count} 个已存在，本次未新增。"
            ),
        }

    if vector_db is not None:
        vector_db.add_documents(documents=new_chunks, ids=new_ids)
    else:
        vector_db = Chroma.from_documents(
            documents=new_chunks,
            embedding=get_embeddings(),
            ids=new_ids,
            persist_directory=resolved_path,
            collection_name="my_docs",
        )

    return vector_db, {
        "success": True,
        "message": (
            f"成功解析并入库 {success_count} 个文件。"
            f"共提取 {len(all_chunks)} 个文本块，批次去重后为 {len(final_chunks)} 个，"
            f"过滤已存在 {existed_count} 个，实际新增 {len(new_chunks)} 个。"
        ),
    }


def clear_database(vector_db, persist_path: str | None = None):
    """清空 Chroma 向量库及其持久化目录。"""
    resolved_path = _resolve_persist_path(persist_path)
    if vector_db is not None:
        client = vector_db._client

        try:
            vector_db.delete_collection()
        except Exception:
            pass

        try:
            client._system.stop()
        except Exception:
            pass

        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        gc.collect()
        time.sleep(0.5)

    if os.path.exists(resolved_path):
        try:
            shutil.rmtree(resolved_path)
            return True, "知识库及持久化目录已彻底删除。"
        except Exception:
            for filename in os.listdir(resolved_path):
                file_path = os.path.join(resolved_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception:
                    pass
            return True, "知识库已清空，残留目录可在重启后释放。"

    return False, "知识库不存在。"
