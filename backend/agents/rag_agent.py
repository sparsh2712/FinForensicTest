import os
import gc
import tracemalloc
from typing import Dict, List
from backend.utils.ocr_vector_store import OCRVectorStore
from backend.utils.utils import log_memory_usage, logger
from dotenv import load_dotenv
import yaml
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.utils.prompt_manager import PromptManager

load_dotenv()

# Get current directory and base paths using relative references
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)  # Go up one level from /agents to /backend

# Load LLM config
LLM_CONFIG_PATH = os.path.join(BASE_DIR, "assets", "llm_config.yaml")
try:
    with open(LLM_CONFIG_PATH, 'r') as f:
        LLM_CONFIG = yaml.safe_load(f)
    # Get agent-specific config or fall back to default
    AGENT_CONFIG = LLM_CONFIG.get("rag_agent", LLM_CONFIG.get("default", {}))
except Exception as e:
    print(f"Error loading LLM config: {e}, using defaults")
    AGENT_CONFIG = {"model": "gemini-2.0-flash", "temperature": 0.0}

# Initialize prompt manager
prompt_manager = PromptManager(os.path.join(BASE_DIR, "prompts"))

def summarize_query_results(company: str, query: str, results: List[Dict]) -> str:
    """
    Summarize the RAG results for a specific query.
    
    Args:
        company: The company name
        query: The query that was asked
        results: The retrieval results from the vector store
    
    Returns:
        A concise summary of the information
    """
    print(f"[RAG Agent] Summarizing results for query: {query}")
    
    if not results:
        print(f"[RAG Agent] No results to summarize for query: {query}")
        return "No relevant information found for this query."
    
    try:
        # Initialize LLM using the agent config
        llm = ChatGoogleGenerativeAI(
            model=AGENT_CONFIG["model"],
            temperature=AGENT_CONFIG["temperature"]
        )
        
        # Extract text content from results
        context_texts = []
        for result in results:
            if "text" in result:
                context_texts.append(result["text"])
            else:
                # Handle unexpected result format
                context_texts.append(str(result))
        
        # Join context texts with separators
        context = "\n\n---\n\n".join(context_texts)
        
        # Prepare variables for prompt template
        variables = {
            "company": company,
            "query": query,
            "context": context
        }
        
        # Get prompts from prompt manager
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "rag_agent", 
            "summarize_results", 
            variables
        )
        
        # Create messages for LLM
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        summary = response.content.strip()
        
        print(f"[RAG Agent] Successfully summarized query results, summary length: {len(summary)} chars")
        return summary
        
    except Exception as e:
        print(f"[RAG Agent] Error summarizing query results: {e}")
        return f"Error during summarization: {str(e)}"

def rag_agent(state: Dict) -> Dict:
    print("[RAG Agent] Starting document analysis process...")
    tracemalloc.start()
    log_memory_usage("rag_agent_start")
    
    # Extract configuration from state
    rag_pdf_path = state.get("rag_pdf_path")
    rag_queries = state.get("rag_queries", [])
    company = state.get("company", "Unknown Company")
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
    query_summaries = {}
    
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
                
                # NEW: Generate summary for this query's results
                query_summary = summarize_query_results(company, query, simplified_results)
                query_summaries[query] = query_summary
                
                print(f"[RAG Agent] Query {i+1} complete - Results summarized")
                
                log_memory_usage(f"after_query_{i+1}")
                gc.collect()
                log_memory_usage(f"after_query_{i+1}_gc")
                
            except Exception as e:
                print(f"[RAG Agent] Error processing query '{query}': {e}")
                query_results[query] = [{"error": str(e)}]
                query_summaries[query] = f"Error processing query: {str(e)}"
    
    # Update state with results and summaries
    state["rag_results"] = query_results
    state["rag_summaries"] = query_summaries  # NEW: Add summarized results to state
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