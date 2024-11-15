import logging
from fastapi import APIRouter, HTTPException, Query
import numpy as np
from pydantic import BaseModel
from services.handle_textSearch import handle_text_search

router = APIRouter()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


router = APIRouter()


class TextSearchRequest(BaseModel):
    search_query: str


@router.post("/text-embedding", summary="Search by Text")
async def search_text(request: TextSearchRequest):
    """The search_text endpoint performs a text-based search for similar images.
    It takes a text query as input and returns the embeddings or similar images."""

    search_query = request.search_query

    if not search_query:
        raise HTTPException(
            status_code=400, detail="Search query is required for text search."
        )

    embedding = await handle_text_search(search_query)

    # Perform the image search
    return embedding
