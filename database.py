"""
database.py — ChromaDB vector store setup and MITRE ATT&CK ingestion.

Provides:
- Persistent ChromaDB client and collection (module-level singletons)
- MITRE ATT&CK ingestion pipeline with technique + mitigation chunks
- query_collection() for RAG retrieval
- add_poisoned_document() / delete_document() for attack simulation
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import chromadb
import requests
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import cfg

# ---------------------------------------------------------------------------
# ChromaDB client + collection — module-level singletons
# ---------------------------------------------------------------------------

_client = chromadb.PersistentClient(path=cfg.chroma_db_path)

_collection = _client.get_or_create_collection(
    name=cfg.chroma_collection_name,
    metadata={"hnsw:space": "cosine"},
)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

_embed_doc = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    google_api_key=cfg.google_api_key,
)

_embed_query = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="RETRIEVAL_QUERY",
    google_api_key=cfg.google_api_key,
)

# ---------------------------------------------------------------------------
# MITRE ATT&CK ingestion
# ---------------------------------------------------------------------------

_STIX_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)
_STIX_CACHE = Path("enterprise-attack.json")


def _download_stix() -> Path:
    """Download enterprise-attack.json if not already cached."""
    if _STIX_CACHE.exists():
        print(f"  Using cached STIX file: {_STIX_CACHE}")
        return _STIX_CACHE

    print(f"  Downloading MITRE ATT&CK STIX data from {_STIX_URL} ...")
    response = requests.get(_STIX_URL, timeout=120)
    response.raise_for_status()
    _STIX_CACHE.write_bytes(response.content)
    print(f"  Saved to {_STIX_CACHE} ({len(response.content) / 1_048_576:.1f} MB)")
    return _STIX_CACHE


def _extract_chunks(stix_path: Path) -> tuple[list[str], list[dict], list[str]]:
    """
    Parse MITRE ATT&CK STIX bundle and extract text chunks with metadata.

    Returns (texts, metadatas, ids).
    """
    from mitreattack.stix20 import MitreAttackData

    attack_data = MitreAttackData(str(stix_path))
    techniques = attack_data.get_techniques(remove_revoked_deprecated=True)

    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    collection_date = datetime.now(timezone.utc).isoformat()

    for technique in techniques:
        technique_id = attack_data.get_attack_id(technique.id)
        if not technique_id:
            continue

        name = technique.get("name", "")
        description = technique.get("description", "")
        # Tactics: list of phase names
        kill_chain_phases = technique.get("kill_chain_phases", []) or []
        tactics = [p.get("phase_name", "") for p in kill_chain_phases if isinstance(p, dict)]
        tactics_str = ", ".join(tactics)

        # --- Chunk 1: Technique description ---
        tech_text = (
            f"Technique: {name}\n"
            f"ATT&CK ID: {technique_id}\n"
            f"Tactics: {tactics_str}\n"
            f"Description: {description}"
        )
        tech_id = f"tech_{technique_id.replace('.', '_')}"
        texts.append(tech_text)
        metadatas.append(
            {
                "source_id": technique_id,
                "collection_date": collection_date,
                "is_poisoned": False,
                "chunk_type": "technique_description",
                "technique_name": name,
                "tactics": tactics_str,
            }
        )
        ids.append(tech_id)

        # --- Chunk 2+: Mitigations ---
        try:
            mitigations = attack_data.get_mitigations_mitigating_technique(technique.id)
        except Exception:
            mitigations = []

        for mit_rel in mitigations:
            try:
                mit_obj = attack_data.get_object_by_stix_id(mit_rel.get("source_ref", ""))
                if not mit_obj:
                    continue
                mit_id = attack_data.get_attack_id(mit_obj.id) or mit_obj.id
                mit_name = mit_obj.get("name", "")
                mit_desc = mit_obj.get("description", "")

                mit_text = (
                    f"Mitigation for {name} ({technique_id}):\n"
                    f"Mitigation ID: {mit_id}\n"
                    f"Mitigation Name: {mit_name}\n"
                    f"Description: {mit_desc}"
                )
                mit_chunk_id = f"mit_{technique_id.replace('.', '_')}_{mit_id.replace('.', '_')}"
                texts.append(mit_text)
                metadatas.append(
                    {
                        "source_id": f"{technique_id}:{mit_id}",
                        "collection_date": collection_date,
                        "is_poisoned": False,
                        "chunk_type": "mitigation",
                        "technique_name": name,
                        "tactics": tactics_str,
                    }
                )
                ids.append(mit_chunk_id)
            except Exception:
                continue

    return texts, metadatas, ids


def ingest_mitre_attack(force: bool = False) -> int:
    """
    Download and ingest MITRE ATT&CK into ChromaDB.

    Args:
        force: Re-ingest even if collection is already populated.

    Returns:
        Number of chunks upserted.
    """
    existing = _collection.count()
    if existing > 0 and not force:
        print(f"  Collection already has {existing} documents. Use --force to re-ingest.")
        return existing

    stix_path = _download_stix()
    print("  Parsing MITRE ATT&CK STIX data ...")
    texts, metadatas, ids = _extract_chunks(stix_path)
    print(f"  Extracted {len(texts)} chunks (techniques + mitigations)")

    # Embed in batches of 50 with 0.5s pause
    EMBED_BATCH = 50
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        print(f"  Embedding batch {i // EMBED_BATCH + 1}/{(len(texts) - 1) // EMBED_BATCH + 1} ...")
        embeddings = _embed_doc.embed_documents(batch)
        all_embeddings.extend(embeddings)
        if i + EMBED_BATCH < len(texts):
            time.sleep(0.5)

    # Upsert to ChromaDB in batches of 500
    UPSERT_BATCH = 500
    total_upserted = 0
    for i in range(0, len(texts), UPSERT_BATCH):
        _collection.upsert(
            ids=ids[i : i + UPSERT_BATCH],
            embeddings=all_embeddings[i : i + UPSERT_BATCH],
            documents=texts[i : i + UPSERT_BATCH],
            metadatas=metadatas[i : i + UPSERT_BATCH],
        )
        total_upserted += len(texts[i : i + UPSERT_BATCH])

    print(f"  Upserted {total_upserted} chunks into ChromaDB collection '{cfg.chroma_collection_name}'")
    return total_upserted


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query_collection(
    query_text: str,
    top_k: int = cfg.retrieval_top_k,
    where: Optional[dict] = None,
) -> List[Document]:
    """
    Retrieve top-k most relevant documents for a query.

    Returns LangChain Document objects with relevance_score in metadata.
    Uses RETRIEVAL_QUERY task type for the query embedding.
    """
    query_embedding = _embed_query.embed_query(query_text)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, max(1, _collection.count())),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = _collection.query(**kwargs)

    docs: List[Document] = []
    for doc_text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Cosine distance → similarity score
        relevance_score = 1.0 - distance
        enriched_metadata = dict(metadata)
        enriched_metadata["relevance_score"] = relevance_score
        docs.append(Document(page_content=doc_text, metadata=enriched_metadata))

    return docs


# ---------------------------------------------------------------------------
# Poison injection / cleanup (for attack simulation)
# ---------------------------------------------------------------------------


def add_poisoned_document(doc_text: str, attack_type: str) -> str:
    """
    Insert a poisoned document into the collection.

    Returns the generated document ID for later cleanup.
    """
    doc_id = f"poison_{attack_type}_{int(time.time() * 1000)}"
    embedding = _embed_doc.embed_documents([doc_text])[0]
    collection_date = datetime.now(timezone.utc).isoformat()

    _collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[doc_text],
        metadatas=[
            {
                "source_id": doc_id,
                "collection_date": collection_date,
                "is_poisoned": True,
                "chunk_type": "poisoned",
                "technique_name": f"POISON:{attack_type}",
                "tactics": "adversarial",
            }
        ],
    )
    return doc_id


def delete_document(doc_id: str) -> None:
    """Remove a document from the collection by ID."""
    _collection.delete(ids=[doc_id])
