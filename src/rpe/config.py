"""Project configuration: `config.yaml` (hyperparameters, paths) + `.env` (secrets)."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class DataConfig(BaseModel):
    raw_path: str = "data/raw/arxiv-metadata-oai-snapshot.json"
    processed_dir: str = "data/processed"
    categories: list[str] = Field(default_factory=lambda: ["cs.CL"])
    start_date: date | None = date(2023, 1, 1)
    end_date: date | None = None
    max_papers: int | None = 20000
    sample_dev: int = 1000
    seed: int = 42


class EmbeddingsConfig(BaseModel):
    batch_size: int = 64
    checkpoint_every: int = 2048
    query_instruction: str = "Represent this sentence for searching relevant passages: "


class IndexConfig(BaseModel):
    dir: str = "data/index"


class RetrievalConfig(BaseModel):
    top_k: int = 10
    rrf_k: int = 60
    candidates: int = 100
    rag_top_k: int = 5


class UIConfig(BaseModel):
    conversations_dir: str = "data/conversations"


class CompareConfig(BaseModel):
    max_papers: int = 5
    cache_dir: str = "data/cache/compare"
    pdf_chars: int = 3000


class LLMConfig(BaseModel):
    timeout: float = 120.0
    max_retries: int = 4
    backoff_seconds: float = 2.0
    temperature: float = 0.0
    max_tokens: int = 2048
    max_parse_retries: int = 2
    json_mode: bool = True
    cache_dir: str = "data/cache/llm"


class PdfConfig(BaseModel):
    max_pdfs: int = 300
    delay_seconds: float = 3.0
    dir: str = "data/pdfs"
    chunk_chars: int = 1500
    chunk_overlap: int = 200
    demo_queries: list[str] = Field(default_factory=list)


class Config(BaseModel):
    """Typed view of `config.yaml`; relative paths are resolved against `root`."""

    data: DataConfig = Field(default_factory=DataConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    compare: CompareConfig = Field(default_factory=CompareConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pdfs: PdfConfig = Field(default_factory=PdfConfig)
    root: Path = PROJECT_ROOT

    def path(self, rel: str | Path) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def artifact(self, rel: str | Path, dev: bool = False) -> Path:
        """Path of a build artifact; --dev artifacts get a `_dev` suffix (file or dir)."""
        p = self.path(rel)
        if not dev:
            return p
        return p.with_name(f"{p.stem}_dev{p.suffix}") if p.suffix else p.with_name(f"{p.name}_dev")

    @property
    def raw_path(self) -> Path:
        return self.path(self.data.raw_path)

    @property
    def processed_dir(self) -> Path:
        return self.path(self.data.processed_dir)

    def papers_path(self, dev: bool = False) -> Path:
        """Parquet produced by `prepare-data` (a separate file in --dev mode)."""
        return self.artifact(self.processed_dir / "papers.parquet", dev)

    def index_dir(self, dev: bool = False) -> Path:
        return self.artifact(self.index.dir, dev)


class EnvSettings(BaseModel):
    """Secrets and model names read from the environment / `.env`."""

    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)


def load_config(path: str | Path | None = None) -> Config:
    """Load `config.yaml` (or `$RPE_CONFIG`); missing file -> defaults."""
    cfg_path = Path(path or os.environ.get("RPE_CONFIG", PROJECT_ROOT / "config.yaml"))
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
    return Config(**(raw or {}), root=cfg_path.resolve().parent)


def load_env(env_file: str | Path | None = None) -> EnvSettings:
    """Load `.env` (without overriding real env vars) and return typed settings."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    return EnvSettings(
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL") or None,
        embedding_model=os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached configuration."""
    return load_config()
