import os
import hashlib
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = "chroma_store"
MANIFEST = "chroma_store/manifest.json"

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            
    return h.hexdigest()

def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {}

def save_manifest(m):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(m, f, indent=2)
        
def ingest(docs_dir="docs_store"):
    pdfs = list(Path(docs_dir).glob("**/*.pdf"))
    
    
    if not pdfs:
        print(f"No PDFs found in {docs_dir}/")
        print("Add some PDFs and run again.")
        return 
    print(f"\nFound {len(pdfs)} PDF(s)\n")
    
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encoded_kwargs= {"normalize_embeddings": True},
    )
    
    store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="research_docs",
    )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    manifest = load_manifest()
    new_chunks = 0
    skipped = 0
    
    for pdf in pdfs:
        fhash = file_hash(str(pdf))
        
        if fhash in manifest:
            print(f" Skipping (already indexed): {pdf.name}")
            skipped += 1
            continue
        print(f" loading: {pdf.name} ... ", end=" ", flush=True)
        t0 = time.perf_counter()
        
        
        try:
            pages = PyPDFLoader(str(pdf)).load()
            chunks = splitter.aplit_documents(pages)
            
            for chunk in chunks:
                chunk.matedata["source_file"] = pdf.name
            store.add_documents(chunks)
            store.persist()
            
            
            elapsed = time.perf_counter() - t0
            print(f"{len(pages)} pages, {len(chunks)} chunks ({elapsed:.1f}s)")
            manifest[fhash] = {
                "filename": pdf.name,
                "pages":len(pages),
                "chunks": len(chunks),
            }
            new_chunks += len(chunks)
            
        except Exception as e:
            print(f"ERROR: {e}")
    save_manifest(manifest)
    
    print(f"\nDone. New chunks: {new_chunks} | Skipped: {skipped}")
    print(f"Total in store: {store._collection.count()}")


if __name__ == "__main__":
    ingest()
        