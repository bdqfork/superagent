import logging
from typing import Literal

from decouple import config
from langchain.docstore.document import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from qdrant_client.http.models import PointStruct

from app.utils.helpers import get_first_non_null
from app.vectorstores.abstract import VectorStoreBase

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStoreBase):
    def __init__(
        self,
        options: dict,
        index_name: str = None,
        host: str = None,
        api_key: str = None,
    ) -> None:
        self.options = options

        variables = {
            "QDRANT_INDEX": get_first_non_null(
                index_name,
                options.get("QDRANT_INDEX"),
                config("QDRANT_INDEX", None),
            ),
            "QDRANT_HOST": get_first_non_null(
                host,
                options.get("QDRANT_HOST"),
                config("QDRANT_HOST", None),
            ),
            "QDRANT_API_KEY": get_first_non_null(
                api_key,
                options.get("QDRANT_API_KEY"),
                config("QDRANT_API_KEY", None),
            ),
        }

        for var, value in variables.items():
            if not value:
                raise ValueError(
                    f"Please provide a {var} via the "
                    f"`{var}` environment variable"
                    "or check the `VectorDb` table in the database."
                )

        self.client = QdrantClient(
            url=variables["QDRANT_HOST"],
            api_key=variables["QDRANT_API_KEY"],
        )

        embeddings = config("EMBEDDINGS", "openai")
        if embeddings == "fastembed":
            self.embeddings = FastEmbedEmbeddings(
                model_name=config("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5")
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                model=config("EMBEDDINGS_MODEL", "text-embedding-3-small"),
                openai_api_key=config("OPENAI_API_KEY"),
            )

        self.index_name = variables["QDRANT_INDEX"]
        logger.info(f"Initialized Qdrant Client with: {self.index_name}")

    def embed_documents(self, documents: list[Document], _: int = 100) -> None:
        collections = self.client.get_collections()
        if self.index_name not in [c.name for c in collections.collections]:
            self.client.recreate_collection(
                collection_name=self.index_name,
                vectors_config={
                    "content": rest.VectorParams(
                        distance=rest.Distance.COSINE,
                        size=config("EMBEDDINGS_SIZE", 1536),
                    ),
                },
            )
        points = []
        i = 0
        for document in documents:
            i += 1
            embeddings = self.embeddings.embed_query(document.page_content)
            points.append(
                PointStruct(
                    id=i,
                    vector={"content": embeddings},
                    payload={"text": document.page_content, **document.metadata},
                )
            )
        self.client.upsert(collection_name=self.index_name, wait=True, points=points)

    def query_documents(
        self,
        prompt: str,
        datasource_id: str,
        top_k: int | None,
        _query_type: Literal["document", "all"] = "document",
    ) -> list[str]:
        embeddings = self.embeddings.embed_query(prompt)
        search_result = self.client.search(
            collection_name=self.index_name,
            query_vector=("content", embeddings),
            limit=top_k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="datasource_id",
                        match=models.MatchValue(value=datasource_id),
                    ),
                ]
            ),
            with_payload=True,
        )
        return search_result

    def delete(self, datasource_id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.index_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="datasource_id",
                                match=models.MatchValue(value=datasource_id),
                            ),
                        ],
                    )
                ),
            )
        except Exception as e:
            logger.error(f"Failed to delete {datasource_id}. Error: {e}")
