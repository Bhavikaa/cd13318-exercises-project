import chromadb
import os
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    chroma_dirs = [
        path for path in current_dir.rglob("*")
        if path.is_dir()
        and "chroma" in path.name.lower()
        and ((path / "chroma.sqlite3").exists() or any(path.glob("*.sqlite3")))
    ]

    for directory in sorted(chroma_dirs):
        try:
            client = chromadb.PersistentClient(
                path=str(directory),
                settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()

            for collection in collections:
                collection_name = getattr(collection, "name", str(collection))
                try:
                    collection_obj = client.get_collection(collection_name)
                    count = collection_obj.count()
                except Exception:
                    count = "unknown"

                key = f"{directory.name}:{collection_name}"
                backends[key] = {
                    "directory": str(directory),
                    "collection_name": collection_name,
                    "display_name": f"{collection_name} ({directory}, {count} docs)",
                    "document_count": str(count)
                }
        except Exception as e:
            error_text = str(e)[:80]
            key = f"{directory.name}:error"
            backends[key] = {
                "directory": str(directory),
                "collection_name": "",
                "display_name": f"{directory} (unavailable: {error_text})",
                "document_count": "0",
                "error": error_text
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        openai_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        embedding_function = None
        if openai_key:
            embedding_function = OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small",
                api_base="https://openai.vocareum.com/v1" if openai_key.startswith("voc") else None
            )
        collection = client.get_collection(collection_name, embedding_function=embedding_function)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None
    if mission_filter and mission_filter.lower() not in {"all", "any", "none"}:
        where_filter = {"mission": mission_filter}

    query_kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    # Chroma returns results sorted by distance. Remove repeated snippets while
    # preserving that score order so the LLM receives varied evidence.
    if results and results.get("documents"):
        for result_index, documents in enumerate(results["documents"]):
            seen = set()
            kept_positions = []
            for position, document in enumerate(documents):
                normalized = " ".join(document.split())
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    kept_positions.append(position)

            for key in ["documents", "metadatas", "distances", "ids"]:
                if key in results and len(results[key]) > result_index:
                    results[key][result_index] = [
                        results[key][result_index][position]
                        for position in kept_positions
                    ]

    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    context_parts = ["Relevant NASA mission document excerpts:"]

    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        metadata = metadata or {}
        mission = metadata.get("mission", "unknown").replace("_", " ").title()
        source = metadata.get("source", "unknown source")
        category = metadata.get("document_category", "document").replace("_", " ").title()

        excerpt = document.strip()
        if len(excerpt) > 1500:
            excerpt = excerpt[:1500].rsplit(" ", 1)[0] + "..."

        context_parts.append(f"\n[Source {index}: {mission} | {category} | {source}]")
        context_parts.append(excerpt)

    return "\n".join(context_parts)
