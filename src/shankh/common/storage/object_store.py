class ObjectStore:
    def __init__(self, bucket: str, prefix: str = "") -> None:
        self._bucket = bucket
        self._prefix = prefix

    def put(self, key: str, data: bytes) -> str:
        ...

    def get(self, key: str) -> bytes:
        ...
