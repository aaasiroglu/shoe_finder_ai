import argparse
import json
from typing import Optional, Dict, Any
from chromadb import PersistentClient

def get_chroma_client():
    """
    Initialize and return a ChromaDB PersistentClient.
    """
    try:
        client = PersistentClient(path="./chroma_db")
        return client
    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}")
        return None


def get_shoe_collection(client):
    """
    Get the shoe collection from the ChromaDB client. If it doesn't exist, create it.
    """
    try:
        collection = client.get_or_create_collection(name="shoe_images")
        return collection
    except Exception as e:
        print(f"Error getting or creating shoe collection: {e}")
        return None


def print_all_records(collection):
    """
    Print all records in the collection with their ids, metadatas, and documents.
    """
    try:
        results = collection.get(include=["metadatas", "documents"])
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        total = len(ids)
        if total == 0:
            print("No records found in the collection.")
            return
        print(f"Total records: {total}\n")
        for idx, (id_, metadata, doc) in enumerate(zip(ids, metadatas, documents)):
            print(f"--- Record {idx+1} ---")
            print(f"ID: {id_}")
            print(f"Metadata: {metadata}")
            print(f"Document: {doc}\n")
    except Exception as e:
        print(f"Error fetching records: {e}")


def print_count(collection):
    try:
        cnt = collection.count()
        print(f"Count: {cnt}")
    except Exception as e:
        print(f"Error counting records: {e}")


def print_head(collection, n: int = 5):
    try:
        results = collection.get(include=["metadatas", "documents"])
        ids = results.get("ids", [])[:n]
        metadatas = results.get("metadatas", [])[:n]
        documents = results.get("documents", [])[:n]
        total = len(ids)
        if total == 0:
            print("No records found in the collection.")
            return
        print(f"Showing first {total} records:\n")
        for idx, (id_, metadata, doc) in enumerate(zip(ids, metadatas, documents), start=1):
            print(f"--- Record {idx} ---")
            print(f"ID: {id_}")
            print(f"Metadata: {metadata}")
            print(f"Document: {doc}\n")
    except Exception as e:
        print(f"Error fetching head: {e}")


def filter_records(collection, where: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
    try:
        kwargs = {"include": ["metadatas", "documents"]}
        if where:
            kwargs["where"] = where
        results = collection.get(**kwargs)
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        if limit is not None:
            ids = ids[:limit]
            metadatas = metadatas[:limit]
            documents = documents[:limit]
        total = len(ids)
        if total == 0:
            print("No matching records found.")
            return
        print(f"Matched records: {total}\n")
        for idx, (id_, metadata, doc) in enumerate(zip(ids, metadatas, documents), start=1):
            print(f"--- Record {idx} ---")
            print(f"ID: {id_}")
            print(f"Metadata: {metadata}")
            print(f"Document: {doc}\n")
    except Exception as e:
        print(f"Error filtering records: {e}")


def search_vector(collection, query: str, top_k: int = 5):
    """Proxy to utils.vector_search_shoes for convenience."""
    try:
        import utils
        result = utils.vector_search_shoes(query, top_k)
        print(result)
    except Exception as e:
        print(f"Error running vector search: {e}")


def main():
    parser = argparse.ArgumentParser(description="ChromaDB Shoe Collection Utilities")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--count", action="store_true", help="Print total record count")
    group.add_argument("--list", action="store_true", help="List all records")
    group.add_argument("--head", type=int, help="Show first N records")
    group.add_argument("--filter", type=str, help="Filter with JSON where clause, e.g. '{\"shoe_type\": \"spor\"}'")
    group.add_argument("--search", type=str, help="Vector search with natural language query")
    parser.add_argument("--top_k", type=int, default=5, help="Top K for vector search")
    parser.add_argument("--limit", type=int, help="Limit number of records printed for list/filter")

    args = parser.parse_args()
    if not (args.count or args.list or args.head is not None or args.filter or args.search):
        args.count = True

    client = get_chroma_client()
    if client is None:
        return
    collection = get_shoe_collection(client)
    if collection is None:
        return

    if args.count:
        print_count(collection)
    elif args.list:
        if args.limit:
            print_head(collection, args.limit)
        else:
            print_all_records(collection)
    elif args.head is not None:
        print_head(collection, args.head)
    elif args.filter:
        try:
            where = json.loads(args.filter)
            filter_records(collection, where=where, limit=args.limit)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON for --filter: {e}")
    elif args.search:
        search_vector(collection, args.search, top_k=args.top_k)

if __name__ == "__main__":
    main()
