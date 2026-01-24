from fastapi import FastAPI, Query
from pydantic import BaseModel
from src.search import text_search, compound_search

app = FastAPI(title="BioSearch API")

class SearchResponse(BaseModel):
    papers: list
    compounds: list
    query: str

@app.get("/search/text")
async def search_by_text(query: str = Query(..., min_length=3)):
    results = text_search(query)
    return SearchResponse(
        papers=results["papers"],
        compounds=results["compounds"],
        query=query
    )

@app.get("/search/compound")
async def search_by_smiles(smiles: str = Query(...)):
    results = compound_search(smiles)
    return results