"""
database.py — ChromaDB vector store setup and MITRE ATT&CK ingestion.

Provides:
- get_or_create_collection() for lazy collection access
- inject_poisoned_document() / remove_poisoned_document() for attack simulation
- query_knowledge_base() for RAG retrieval (returns plain dicts)
- ingest_mitre_attack() for MITRE ATT&CK ingestion
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import cfg

_embed_doc: Optional[GoogleGenerativeAIEmbeddings] = None
_embed_query: Optional[GoogleGenerativeAIEmbeddings] = None


def _get_embed_doc() -> GoogleGenerativeAIEmbeddings:
    global _embed_doc
    if _embed_doc is None:
        _embed_doc = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            google_api_key=cfg.google_api_key,
        )
    return _embed_doc


def _get_embed_query() -> GoogleGenerativeAIEmbeddings:
    global _embed_query
    if _embed_query is None:
        _embed_query = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="RETRIEVAL_QUERY",
            google_api_key=cfg.google_api_key,
        )
    return _embed_query


def get_or_create_collection(
    db_path: str = cfg.chroma_db_path,
    name: str = cfg.chroma_collection_name,
):
    """Lazily create a ChromaDB PersistentClient and return the collection."""
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def inject_poisoned_document(collection, doc_id: str, payload_text: str, metadata: dict) -> None:
    """Insert a poisoned document into the given collection."""
    embedding = _get_embed_doc().embed_documents([payload_text])[0]
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[payload_text],
        metadatas=[metadata],
    )


def remove_poisoned_document(collection, doc_id: str) -> None:
    """Remove a document from the collection by ID."""
    collection.delete(ids=[doc_id])


def query_knowledge_base(collection, query: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve top-n most relevant documents for a query.

    Returns list of dicts: [{page_content, metadata, relevance_score}, ...]
    Uses RETRIEVAL_QUERY task type for the query embedding.
    """
    query_embedding = _get_embed_query().embed_query(query)

    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc_text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        relevance_score = 1.0 - distance
        docs.append({
            "page_content": doc_text,
            "metadata": dict(metadata),
            "relevance_score": relevance_score,
        })

    return docs

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

    import requests
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
        kill_chain_phases = technique.get("kill_chain_phases", []) or []
        tactics = [p.get("phase_name", "") for p in kill_chain_phases if isinstance(p, dict)]
        tactics_str = ", ".join(tactics)

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


def ingest_mitre_attack(collection, force: bool = False) -> int:
    """
    Download and ingest MITRE ATT&CK into ChromaDB.

    Args:
        collection: ChromaDB collection to ingest into.
        force: Re-ingest even if collection is already populated.

    Returns:
        Number of chunks upserted.
    """
    existing = collection.count()
    if existing > 0 and not force:
        print(f"  Collection already has {existing} documents. Use --force to re-ingest.")
        return existing

    stix_path = _download_stix()
    print("  Parsing MITRE ATT&CK STIX data ...")
    texts, metadatas, ids = _extract_chunks(stix_path)
    print(f"  Extracted {len(texts)} chunks (techniques + mitigations)")

    EMBED_BATCH = 50
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        print(f"  Embedding batch {i // EMBED_BATCH + 1}/{(len(texts) - 1) // EMBED_BATCH + 1} ...")
        embeddings = _get_embed_doc().embed_documents(batch)
        all_embeddings.extend(embeddings)
        if i + EMBED_BATCH < len(texts):
            time.sleep(0.5)

    UPSERT_BATCH = 500
    total_upserted = 0
    for i in range(0, len(texts), UPSERT_BATCH):
        collection.upsert(
            ids=ids[i : i + UPSERT_BATCH],
            embeddings=all_embeddings[i : i + UPSERT_BATCH],
            documents=texts[i : i + UPSERT_BATCH],
            metadatas=metadatas[i : i + UPSERT_BATCH],
        )
        total_upserted += len(texts[i : i + UPSERT_BATCH])

    print(f"  Upserted {total_upserted} chunks into ChromaDB collection '{collection.name}'")
    return total_upserted
