import json
import os
import time
import logging
import tempfile
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("rag_agent")

class SimpleTextChunk:
    """Simple class for text chunks with metadata"""
    
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}
        self.embedding = None
    
    def to_dict(self):
        return {
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding
        }

class SimpleVectorStore:
    """Simple in-memory vector store for text similarity search"""
    
    def __init__(self):
        self.chunks = []
        self.vectorizer = CountVectorizer(stop_words="english")
        self.document_term_matrix = None
        self.initialized = False
    
    def add_chunks(self, chunks: List[SimpleTextChunk]):
        """Add text chunks to the vector store"""
        if not chunks:
            return
            
        self.chunks.extend(chunks)
        self._update_embeddings()
    
    def _update_embeddings(self):
        """Update the document-term matrix"""
        texts = [chunk.text for chunk in self.chunks]
        self.document_term_matrix = self.vectorizer.fit_transform(texts)
        self.initialized = True
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks to the query"""
        if not self.initialized or not self.chunks:
            return []
            
        # Convert query to vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_term_matrix).flatten()
        
        # Get top k indices
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": float(similarities[idx])
            })
        
        return results

class SimpleDocumentProcessor:
    """Simple document processor for PDF documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_pdf(self, pdf_path: str) -> List[SimpleTextChunk]:
        """Process a PDF document and return text chunks"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []
            
        try:
            doc = fitz.open(pdf_path)
            chunks = []
            
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text()
                
                if not text.strip():
                    continue
                
                page_chunks = self._chunk_text(text, {
                    "source": os.path.basename(pdf_path),
                    "page": page_idx + 1
                })
                
                chunks.extend(page_chunks)
            
            doc.close()
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return []
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[SimpleTextChunk]:
        """Split text into chunks with overlap"""
        if not text:
            return []
            
        chunks = []
        
        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [SimpleTextChunk(text=text, metadata=metadata.copy())]
        
        # Otherwise, split into overlapping chunks
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks)
            
            end = min(i + self.chunk_size, len(text))
            chunk_text = text[i:end]
            
            if not chunk_text.strip():
                continue
                
            chunks.append(SimpleTextChunk(text=chunk_text, metadata=chunk_metadata))
        
        return chunks

def rag_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified RAG (Retrieval-Augmented Generation) agent that processes PDFs
    and answers queries based on document content.
    
    Args:
        state: The current state dictionary containing:
            - company: Company name
            - rag_pdf_path: Path to the uploaded PDF
            - rag_queries: List of queries to answer using the document
    
    Returns:
        Updated state containing RAG analysis results and next routing information
    """
    logger.info(f"Starting RAG agent for {state.get('company')}")
    
    try:
        company = state.get("company", "")
        pdf_path = state.get("rag_pdf_path")
        queries = state.get("rag_queries", [])
        
        if not company:
            logger.error("Company name is missing!")
            return {**state, "goto": "meta_agent", "rag_status": "ERROR", "error": "Company name is missing"}
        
        if not pdf_path or not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return {**state, "goto": "meta_agent", "rag_status": "ERROR", "error": f"PDF file not found: {pdf_path}"}
        
        if not queries:
            # Use default queries if none provided
            queries = [
                f"What are the key financial metrics for {company}?",
                f"Are there any regulatory or compliance issues mentioned for {company}?",
                f"What is the corporate governance structure of {company}?",
                f"What are the main risks identified for {company}?"
            ]
        
        logger.info(f"Processing PDF: {pdf_path} with {len(queries)} queries")
        
        # Process the PDF
        doc_processor = SimpleDocumentProcessor()
        chunks = doc_processor.process_pdf(pdf_path)
        
        if not chunks:
            logger.error("No text extracted from PDF")
            return {**state, "goto": "meta_agent", "rag_status": "ERROR", "error": "No text extracted from PDF"}
        
        logger.info(f"Extracted {len(chunks)} text chunks from PDF")
        
        # Add chunks to vector store
        vector_store = SimpleVectorStore()
        vector_store.add_chunks(chunks)
        
        # Process queries
        query_results = []
        for query in queries:
            logger.info(f"Processing query: {query}")
            
            try:
                # Search for relevant chunks
                search_results = vector_store.search(query, k=5)
                
                if not search_results:
                    logger.warning(f"No relevant content found for query: {query}")
                    query_results.append({
                        "query": query,
                        "answer": "No relevant information found in the document.",
                        "sources": []
                    })
                    continue
                
                # Generate answer using LLM
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
                
                system_prompt = """
                You are a professional, helpful assistant that provides accurate information based on retrieved content.
                Analyze the retrieved text chunks and create a clear, concise, and informative response to the user's query.
                
                1. Focus on answering ONLY with information from the provided chunks
                2. If the chunks don't contain relevant information to answer the query, acknowledge this
                3. NEVER make up information not present in the chunks
                4. ALWAYS cite the specific source and page number for key information
                5. Organize information logically and format it for readability
                """
                
                context = "\n\n".join([
                    f"[Document: {result['metadata'].get('source', 'unknown')}, "
                    f"Page: {result['metadata'].get('page', 'unknown')}]\n"
                    f"{result['text']}"
                    for result in search_results
                ])
                
                human_prompt = f"""
                QUERY: {query}
                
                RETRIEVED CHUNKS ({len(search_results)} sources):
                {context}
                
                Please provide a detailed answer to the query using only the information in the retrieved chunks.
                If the information isn't sufficient to answer completely, let me know what's missing.
                """
                
                messages = [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
                
                response = llm.invoke(messages)
                
                # Add result
                query_results.append({
                    "query": query,
                    "answer": response.content,
                    "sources": [
                        {
                            "source": result["metadata"].get("source", "unknown"),
                            "page": result["metadata"].get("page", "unknown"),
                            "score": result["score"]
                        }
                        for result in search_results
                    ]
                })
                
                logger.info(f"Generated answer for query: {query}")
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                query_results.append({
                    "query": query,
                    "answer": f"Error processing query: {str(e)}",
                    "sources": []
                })
        
        # Generate a summary of findings
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            
            system_prompt = """
            You are a financial analyst summarizing key findings from a document analysis.
            Based on the queries and answers provided, create a comprehensive summary of the most important insights.
            
            Focus on:
            1. Key financial metrics and performance indicators
            2. Governance structure and any issues or red flags
            3. Regulatory compliance and potential issues
            4. Risk factors and their potential impact
            
            Format your response as a JSON object with the following fields:
            - key_insights: List of the most important findings
            - risk_factors: List of identified risks
            - governance_assessment: Brief assessment of governance structure
            - regulatory_compliance: Summary of compliance status and issues
            """
            
            human_prompt = f"""
            Company: {company}
            Document: {os.path.basename(pdf_path)}
            
            Query Results:
            {json.dumps(query_results, indent=2)}
            
            Provide a comprehensive summary of the key findings from the document analysis.
            """
            
            messages = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = llm.invoke(messages)
            response_content = response.content.strip()
            
            if "```json" in response_content:
                json_content = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_content = response_content.split("```")[1].strip()
            else:
                json_content = response_content
                
            summary = json.loads(json_content)
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            summary = {
                "key_insights": ["Summary generation failed due to technical error"],
                "risk_factors": ["Unable to identify risk factors due to analysis error"],
                "governance_assessment": "Assessment unavailable due to technical issues",
                "regulatory_compliance": "Compliance analysis failed due to technical error"
            }
        
        # Prepare the results
        rag_results = {
            "document": os.path.basename(pdf_path),
            "queries": query_results,
            "summary": summary,
            "chunk_count": len(chunks),
            "timestamp": datetime.now().isoformat()
        }
        
        # Update state with results
        state["rag_results"] = rag_results
        state["rag_status"] = "DONE"
        
        # If synchronous_pipeline is set, use the next_agent value, otherwise go to meta_agent
        goto = "meta_agent"
        if state.get("synchronous_pipeline", False):
            goto = state.get("next_agent", "meta_agent")
        
        logger.info(f"RAG agent completed successfully for {company}")
        return {**state, "goto": goto}
    
    except Exception as e:
        logger.error(f"Error in RAG agent: {str(e)}")
        return {
            **state,
            "goto": "meta_agent",
            "rag_status": "ERROR",
            "error": f"Error in RAG agent: {str(e)}"
        }