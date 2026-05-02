from fastapi import APIRouter, HTTPException
from app.ingest.download_novels import download_novels
from app.ingest.process_novels import process_all_novels
from app.ingest.chunk_documents import prepare_chunks
from app.vectorstore.ingest import reingest

router = APIRouter()

@router.post("/download")
def download_novels_api():
    try:
        download_novels()
        return {
            "status": "success",
            "message": "Novels downloaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/process")
def process_all_novels_api():
    try:
        process_all_novels()
        return {
            "status": "success",
            "message": "Novels processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/make-chunks")
def prepare_chunks_api():
    try:
        prepare_chunks()
        return {
            "status": "success",
            "message": "Chunks processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/embed-store")
def embed_and_store_api():
    try:
        count = reingest()
        return {
            "status": "success",
            "message": f"Embedding vectors rebuilt successfully. Collection now has {count} documents."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
    