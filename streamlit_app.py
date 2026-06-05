from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs) -> bool:
        """Fallback when python-dotenv is not installed."""
        return False

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import LOCAL_EMBEDDING_MODEL, LocalEmbedder, _mock_embed
from src.markdown_processor import (
    MarkdownMetadata,
    MarkdownSection,
    normalize_whitespace,
    parse_markdown_metadata,
    sliding_window_chunk,
    split_markdown_sections,
)
from src.models import Document
from src.store import EmbeddingStore


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "dataset"
ENV_PATH = BASE_DIR / ".env"
CHUNKING_STRATEGIES = [
    "Markdown sections + sliding window",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
]

load_dotenv(ENV_PATH, override=False)


def list_data_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return supported data files for the Streamlit UI."""
    if not data_dir.exists():
        return []
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )


def mock_llm(prompt: str) -> str:
    """Small demo LLM used by the chunking lab page."""
    preview = prompt[:900].replace("\n", " ")
    return f"[Demo answer]\n\nPrompt preview:\n{preview}..."


def get_configured_api_key() -> str:
    """Read Gemini API key from .env, Streamlit secrets, or environment variables."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")


def get_configured_model() -> str:
    """Read Gemini model name from .env, Streamlit secrets, or use a default."""
    try:
        secret_model = st.secrets.get("GEMINI_MODEL", "")
    except Exception:
        secret_model = ""
    return secret_model or os.getenv("GEMINI_MODEL", "") or "gemini-2.5-flash"


def make_gemini_llm(api_key: str, model_name: str) -> Callable[[str], str]:
    """
    Create a Gemini-backed LLM function.

    This supports the newer `google-genai` SDK first, then falls back to the
    older `google-generativeai` package if that is what the machine has.
    """
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        def generate_with_genai(prompt: str) -> str:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text or ""

        return generate_with_genai
    except ImportError:
        pass

    try:
        import google.generativeai as genai_old

        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel(model_name)

        def generate_with_old_genai(prompt: str) -> str:
            response = model.generate_content(prompt)
            return getattr(response, "text", "")

        return generate_with_old_genai
    except ImportError as exc:
        raise RuntimeError("Install google-genai or google-generativeai to use Gemini.") from exc


def file_metadata(path: Path, content: str) -> MarkdownMetadata:
    """Parse markdown metadata or provide defaults for plain text files."""
    if path.suffix.lower() == ".md":
        parsed = parse_markdown_metadata(content)
        return MarkdownMetadata(
            title=parsed.title or path.stem,
            url=parsed.url,
            original_price=parsed.original_price,
            current_price=parsed.current_price,
        )
    return MarkdownMetadata(title=path.stem, url="", original_price="", current_price="")


def file_sections(path: Path, content: str) -> list[MarkdownSection]:
    """Return markdown sections, or one full-document section for text files."""
    if path.suffix.lower() == ".md":
        sections = split_markdown_sections(content)
        if sections:
            return sections
    return [MarkdownSection(title="Full document", content=content)]


def chunk_section(
    section_text: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    max_sentences: int,
) -> list[str]:
    """Chunk a section with the selected strategy."""
    if strategy == "Markdown sections + sliding window":
        return sliding_window_chunk(section_text, chunk_size=chunk_size, overlap=overlap)
    if strategy == "FixedSizeChunker":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap).chunk(section_text)
    if strategy == "SentenceChunker":
        return SentenceChunker(max_sentences_per_chunk=max_sentences).chunk(section_text)
    if strategy == "RecursiveChunker":
        return RecursiveChunker(chunk_size=chunk_size).chunk(section_text)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def build_documents(
    paths: list[Path],
    strategy: str,
    chunk_size: int,
    overlap: int,
    max_sentences: int,
) -> list[Document]:
    """Build RAG-ready Document objects from selected files."""
    documents: list[Document] = []

    for path in paths:
        content = path.read_text(encoding="utf-8")
        metadata = file_metadata(path, content)
        sections = file_sections(path, content)

        chunk_index = 0
        for section in sections:
            section_text = normalize_whitespace(section.content)
            chunks = chunk_section(
                section_text=section_text,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                max_sentences=max_sentences,
            )

            for chunk_text in chunks:
                documents.append(
                    Document(
                        id=f"{path.stem}-{chunk_index}",
                        content=chunk_text,
                        metadata={
                            "title": metadata.title,
                            "url": metadata.url,
                            "original_price": metadata.original_price,
                            "current_price": metadata.current_price,
                            "section": section.title,
                            "chunk_index": chunk_index,
                            "source": str(path),
                            "chunk_strategy": strategy,
                        },
                    )
                )
                chunk_index += 1

    return documents


def build_store(documents: list[Document], embedding_fn: Callable[[str], list[float]]) -> EmbeddingStore:
    """Create an embedding store and add all chunk documents."""
    store = EmbeddingStore(collection_name="streamlit_demo", embedding_fn=embedding_fn)
    store.add_documents(documents)
    return store


@st.cache_resource(show_spinner="Loading embedding model...")
def load_local_embedder(model_name: str) -> LocalEmbedder:
    """Load and cache a local sentence-transformers embedder."""
    return LocalEmbedder(model_name=model_name)


def get_embedding_fn(backend: str, model_name: str) -> Callable[[str], list[float]]:
    """Return the selected embedding function."""
    if backend == "sentence-transformers":
        return load_local_embedder(model_name)
    return _mock_embed


def render_result_card(result: dict[str, Any], index: int) -> None:
    """Render one retrieved chunk result."""
    metadata = result.get("metadata", {})
    title = metadata.get("title", "Untitled")
    section = metadata.get("section", "Unknown section")
    score = result.get("score", 0.0)

    with st.expander(f"{index}. {title} | {section} | score={score:.3f}", expanded=index == 1):
        st.caption(metadata.get("source", ""))
        st.write(result.get("content", ""))
        if metadata.get("url"):
            st.link_button("Open source", metadata["url"])


def render_sidebar(files: list[Path]) -> tuple[str, list[Path], str, int, int, int, int, str, str, str, str]:
    """Render shared controls for both pages."""
    with st.sidebar:
        page = st.radio("Page", ["Chatbot", "Chunking Lab"])

        st.header("Data")
        selected_names = st.multiselect(
            "Choose files",
            options=[path.name for path in files],
            default=[path.name for path in files[:3]],
        )

        st.header("Chunking")
        strategy = st.selectbox("Method", CHUNKING_STRATEGIES)
        chunk_size = st.slider("Chunk size", min_value=100, max_value=2000, value=500, step=50)
        overlap = st.slider("Overlap", min_value=0, max_value=500, value=50, step=10)
        max_sentences = st.slider("Max sentences per chunk", min_value=1, max_value=10, value=3)
        top_k = st.slider("Top K", min_value=1, max_value=10, value=3)

        st.header("Embedding")
        embedding_backend = st.selectbox(
            "Backend",
            ["sentence-transformers", "mock"],
            help="Use sentence-transformers for real semantic retrieval. Mock is only for tests/demo fallback.",
        )
        embedding_model = st.text_input(
            "Embedding model",
            value=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL),
        )

        st.header("Gemini")
        api_key = st.text_input(
            "API key",
            value=get_configured_api_key(),
            type="password",
            help="Can also be set as GEMINI_API_KEY or GOOGLE_API_KEY.",
        )
        model_name = st.text_input("Model", value=get_configured_model())

    selected_paths = [path for path in files if path.name in selected_names]
    return (
        page,
        selected_paths,
        strategy,
        chunk_size,
        overlap,
        max_sentences,
        top_k,
        api_key,
        model_name,
        embedding_backend,
        embedding_model,
    )


def render_metrics(selected_paths: list[Path], documents: list[Document], strategy: str) -> None:
    """Render a short summary of the current RAG index."""
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Files", len(selected_paths))
    col_b.metric("Chunks", len(documents))
    col_c.metric("Strategy", strategy)


def render_chatbot_page(
    store: EmbeddingStore,
    selected_paths: list[Path],
    documents: list[Document],
    strategy: str,
    top_k: int,
    api_key: str,
    model_name: str,
) -> None:
    """Render page 1: a Gemini-powered RAG chatbot."""
    st.title("RAG Chatbot")
    render_metrics(selected_paths, documents, strategy)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chào bạn, mình sẽ trả lời dựa trên các tài liệu đang được chọn.",
            }
        ]

    if st.button("Clear chat"):
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Hỏi về giá, điều khoản, hoàn huỷ, cách sử dụng voucher...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            llm_fn = make_gemini_llm(api_key=api_key, model_name=model_name)
            agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
            answer = agent.answer(question, top_k=top_k)
        except Exception as exc:
            answer = (
                "Không gọi được Gemini. Kiểm tra API key, model name, và Gemini SDK.\n\n"
                f"Chi tiết lỗi: {exc}"
            )

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.subheader("Sources")
        for index, result in enumerate(store.search(question, top_k=top_k), start=1):
            render_result_card(result, index)


def render_chunking_lab_page(
    store: EmbeddingStore,
    selected_paths: list[Path],
    documents: list[Document],
    strategy: str,
    top_k: int,
) -> None:
    """Render page 2: the original chunking/search inspection UI."""
    st.title("Chunking Lab")
    render_metrics(selected_paths, documents, strategy)

    tab_search, tab_chunks = st.tabs(["Ask", "Chunk Preview"])

    with tab_search:
        question = st.text_input(
            "Question",
            value="Voucher có chính sách hoàn huỷ như thế nào?",
        )

        if question:
            results = store.search(question, top_k=top_k)
            agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm)
            answer = agent.answer(question, top_k=top_k)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Chunks")
            for index, result in enumerate(results, start=1):
                render_result_card(result, index)

    with tab_chunks:
        preview_count = st.slider(
            "Preview chunks",
            min_value=1,
            max_value=min(20, len(documents)),
            value=min(5, len(documents)),
        )
        for document in documents[:preview_count]:
            metadata = document.metadata
            with st.expander(f"{document.id} | {metadata.get('section', '')}"):
                st.caption(metadata.get("source", ""))
                st.write(document.content)


def main() -> None:
    """Run the Streamlit RAG demo UI."""
    st.set_page_config(page_title="Markdown RAG Demo", layout="wide")

    files = list_data_files()
    if not files:
        st.warning("No .md or .txt files found in the data folder.")
        return

    (
        page,
        selected_paths,
        strategy,
        chunk_size,
        overlap,
        max_sentences,
        top_k,
        api_key,
        model_name,
        embedding_backend,
        embedding_model,
    ) = render_sidebar(files)

    if overlap >= chunk_size and strategy in {"Markdown sections + sliding window", "FixedSizeChunker"}:
        st.error("Overlap must be smaller than chunk size.")
        return

    if not selected_paths:
        st.info("Choose at least one file to build the RAG index.")
        return

    documents = build_documents(
        selected_paths,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        max_sentences=max_sentences,
    )
    if not documents:
        st.warning("No chunks were created from the selected files and chunking settings.")
        return

    try:
        embedding_fn = get_embedding_fn(embedding_backend, embedding_model)
        store = build_store(documents, embedding_fn=embedding_fn)
    except Exception as exc:
        st.error(
            "Could not load the selected embedding model. "
            "Check that sentence-transformers is installed and the model name is correct."
        )
        st.exception(exc)
        return

    if page == "Chatbot":
        render_chatbot_page(
            store=store,
            selected_paths=selected_paths,
            documents=documents,
            strategy=strategy,
            top_k=top_k,
            api_key=api_key,
            model_name=model_name,
        )
    else:
        render_chunking_lab_page(
            store=store,
            selected_paths=selected_paths,
            documents=documents,
            strategy=strategy,
            top_k=top_k,
        )


if __name__ == "__main__":
    main()
