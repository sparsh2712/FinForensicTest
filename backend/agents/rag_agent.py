import os
import gc
import tracemalloc
from typing import Dict
from backend.utils.ocr_vector_store import OCRVectorStore
from backend.utils.utils import log_memory_usage, logger
from dotenv import load_dotenv

load_dotenv()

def rag_agent(state: Dict) -> Dict:
    print("[RAG Agent] Starting document analysis process...")
    tracemalloc.start()
    log_memory_usage("rag_agent_start")
    
    # Extract configuration from state
    rag_pdf_path = state.get("rag_pdf_path")
    rag_queries = state.get("rag_queries", [])
    is_file_embedded = state.get("is_file_embedded", False)
    vector_store_dir = "vector_stores"
    index_type = "Flat"
    chunk_size = 10000
    chunk_overlap = 500
    
    # Validate inputs
    if not rag_pdf_path:
        print("[RAG Agent] ERROR: PDF path is missing!")
        return {**state, "goto": "meta_agent", "rag_status": "ERROR", "error": "PDF path is required"}
    
    if not os.path.exists(rag_pdf_path):
        print(f"[RAG Agent] ERROR: PDF file not found at {rag_pdf_path}")
        return {**state, "goto": "meta_agent", "rag_status": "ERROR", "error": f"PDF file not found at {rag_pdf_path}"}
    
    # Ensure vector store directory exists
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Initialize vector store with configuration
    vector_store = OCRVectorStore(
        index_type=index_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Determine vector store subdirectory
    doc_id = os.path.basename(rag_pdf_path).replace(".", "_")
    doc_vector_dir = os.path.join(vector_store_dir, doc_id)
    
    # Embed document if needed
    if not is_file_embedded:
        try:
            print(f"[RAG Agent] Embedding document: {rag_pdf_path}")
            log_memory_usage("before_embedding")
            
            vector_store.add_document(rag_pdf_path)
            
            log_memory_usage("after_embedding")
            
            # Save vector store
            os.makedirs(doc_vector_dir, exist_ok=True)
            print(f"[RAG Agent] Saving vector store to {doc_vector_dir}")
            vector_store.save(doc_vector_dir)
            
            state["is_file_embedded"] = True
            print(f"[RAG Agent] Document successfully embedded and saved to {doc_vector_dir}")
            
            gc.collect()
            log_memory_usage("after_embedding_gc")
            
        except Exception as e:
            print(f"[RAG Agent] Error embedding document: {e}")
            import traceback
            print(traceback.format_exc())
            return {**state, "goto": "meta_agent", "rag_status": "ERROR", "error": f"Error embedding document: {str(e)}"}
    else:
        # Load existing vector store
        print(f"[RAG Agent] Loading existing document embeddings from {doc_vector_dir}")
        log_memory_usage("before_loading")
        
        try:
            vector_store.load(doc_vector_dir)
            log_memory_usage("after_loading")
            
        except Exception as e:
            print(f"[RAG Agent] Error loading embeddings: {e}")
            return {**state, "goto": "meta_agent", "rag_status": "ERROR", "error": f"Error loading embeddings: {str(e)}"}
    
    # Process queries
    query_results = {}
    if rag_queries:
        print(f"[RAG Agent] Processing {len(rag_queries)} queries")
        
        for i, query in enumerate(rag_queries):
            try:
                print(f"[RAG Agent] Querying ({i+1}/{len(rag_queries)}): {query}")
                log_memory_usage(f"before_query_{i+1}")
                
                results = vector_store.answer_question(query, k=5)
                
                simplified_results = []
                for j, result in enumerate(results):
                    simplified_results.append({
                        "rank": j+1,
                        "text": result["text"],
                        "metadata": {
                            "page": result["metadata"].get("page", "Unknown"),
                            "source": result["metadata"].get("source", "Unknown")
                        },
                        "score": result["score"]
                    })
                    
                    # Print results in console for debugging
                    print(f"\n[{j+1}] Score: {result['score']:.4f}")
                    print(f"Source: {result['metadata'].get('source', 'Unknown')}, Page: {result['metadata'].get('page', 'Unknown')}")
                    print(f"Text: {result['text'][:200]}...")
                
                query_results[query] = simplified_results
                
                log_memory_usage(f"after_query_{i+1}")
                gc.collect()
                log_memory_usage(f"after_query_{i+1}_gc")
                
            except Exception as e:
                print(f"[RAG Agent] Error processing query '{query}': {e}")
                query_results[query] = [{"error": str(e)}]
    
    # Update state with results
    state["rag_results"] = query_results
    state["rag_status"] = "DONE"
    state["rag_vector_store_path"] = doc_vector_dir
    
    # Report memory statistics
    current, peak = tracemalloc.get_traced_memory()
    print(f"[RAG Agent] MEMORY STATS: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
    tracemalloc.stop()
    
    print(f"[RAG Agent] Document analysis complete. Processed {len(query_results)} queries.")
    log_memory_usage("rag_agent_end")
    
    vector_store = None
    gc.collect()
    
    return {**state, "goto": "meta_agent"}