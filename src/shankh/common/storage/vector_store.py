class VectorStore:
    def __init__(self, collection_name: str, persist_dir: str) -> None:
        self._collection_name = collection_name
        self._persist_dir = persist_dir

    def index_documents(self, documents: list[dict]) -> None:
        ...

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        ...
