from typing import Dict, List, Optional, Tuple, Any
import requests
import json
import time
import re
from datetime import datetime
from markdownify import markdownify as md
from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import concurrent.futures
from threading import Lock
import traceback
import queue
import os
import yaml

# from backend.utils.llm_provider import LLMProviderManager
from backend.utils.prompt_manager import PromptManager

from dotenv import load_dotenv
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
    AGENT_CONFIG = LLM_CONFIG.get("analyst_agent", LLM_CONFIG.get("default", {}))
except Exception as e:
    print(f"Error loading LLM config: {e}, using defaults")
    AGENT_CONFIG = {"model": "gemini-2.0-flash", "temperature": 0.2}

# Initialize prompt manager with path relative to current file
prompt_manager = PromptManager(os.path.join(BASE_DIR, "prompts"))

class ThreadSafeKnowledgeBase:
    def __init__(self):
        self.data = {
            "events": {},          
            "entities": {},        
            "relationships": {},   
            "patterns": {},        
            "red_flags": [],       
            "evidence": {},        
            "timeline": [],        
            "sources": {},         
            "metadata": {}         
        }
        self.lock = Lock()
    
    def store_event_insights(self, event: str, insights: List[Dict]) -> None:
        """Thread-safe storage of insights for an event"""
        with self.lock:
            if event not in self.data["events"]:
                self.data["events"][event] = []
            self.data["events"][event].extend(insights)
    
    def add_to_timeline(self, timeline_items: List[Dict]) -> None:
        """Thread-safe addition to timeline"""
        with self.lock:
            self.data["timeline"].extend(timeline_items)
    
    def add_red_flags(self, flags: List[str]) -> None:
        """Thread-safe addition of red flags"""
        with self.lock:
            for flag in flags:
                if flag not in self.data["red_flags"]:
                    self.data["red_flags"].append(flag)
    
    def update_entities(self, entities: Dict) -> None:
        """Thread-safe update of entity information"""
        with self.lock:
            for entity, info in entities.items():
                if entity not in self.data["entities"]:
                    self.data["entities"][entity] = info
                else:
                    self.data["entities"][entity].update(info)
    
    def get_all_data(self) -> Dict:
        """Thread-safe retrieval of all knowledge base data"""
        with self.lock:
            return self.data.copy()

knowledge_base = ThreadSafeKnowledgeBase()

class ProcessingStats:
    def __init__(self):
        self.total_events = 0
        self.total_articles = 0
        self.processed_articles = 0
        self.articles_with_insights = 0
        self.events_with_insights = 0
        self.failed_articles = 0
        self.lock = Lock()
    
    def increment(self, stat_name: str, amount: int = 1) -> None:
        """Thread-safe increment of a statistic"""
        with self.lock:
            if hasattr(self, stat_name):
                setattr(self, stat_name, getattr(self, stat_name) + amount)
    
    def get_stats(self) -> Dict:
        """Thread-safe retrieval of all statistics"""
        with self.lock:
            return {
                "total_events": self.total_events,
                "total_articles": self.total_articles,
                "processed_articles": self.processed_articles,
                "articles_with_insights": self.articles_with_insights,
                "events_with_insights": self.events_with_insights,
                "failed_articles": self.failed_articles,
                "completion_percentage": (
                    round((self.processed_articles / self.total_articles) * 100, 1)
                    if self.total_articles > 0 else 0
                )
            }

processing_stats = ProcessingStats()

progress_queue = queue.Queue()

def fetch_article_content(url: str, max_retries: int = 3, timeout: int = 30) -> Tuple[Optional[str], Optional[dict]]:
    """
    Enhanced article fetching with multiple methods and metadata extraction.
    Returns tuple of (content, metadata).
    """
    progress_queue.put(f"Fetching content from: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    metadata = {
        "source_domain": urlparse(url).netloc,
        "fetch_timestamp": datetime.now().isoformat(),
        "fetch_method": None,
        "content_size": 0,
        "extraction_success": False
    }
    
    try:
        jina_url = "https://r.jina.ai/" + url
        response = requests.get(jina_url, timeout=timeout, headers=headers)
        if response.status_code == 200 and len(response.text) > 500:  
            metadata["fetch_method"] = "jina"
            metadata["content_size"] = len(response.text)
            metadata["extraction_success"] = True
            progress_queue.put(f"Jina extraction successful for: {url}")
            return response.text, metadata
    except Exception as e:
        progress_queue.put(f"Jina extraction failed for {url}: {str(e)[:100]}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            if response.status_code == 200:
                html_content = response.text
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                
                article_content = None
                for selector in ['article', 'main', '.article-content', '.post-content', '#article', '#content', '.content']:
                    content = soup.select_one(selector)
                    if content and len(content.get_text(strip=True)) > 200:
                        article_content = content
                        break
                
                if not article_content:
                    article_content = soup.body
                
                if article_content:
                    markdown_content = md(str(article_content))
                    
                    metadata["fetch_method"] = "requests_with_extraction"
                    metadata["content_size"] = len(markdown_content)
                    metadata["extraction_success"] = True
                    progress_queue.put(f"Direct extraction successful for: {url}")
                    return markdown_content, metadata
                else:
                    progress_queue.put(f"Failed to extract content from {url}")
            else:
                progress_queue.put(f"HTTP error {response.status_code} for {url}")
                
            time.sleep(1)
            
        except Exception as e:
            progress_queue.put(f"Error during attempt {attempt+1} for {url}: {str(e)[:100]}...")
            time.sleep(2)
    
    progress_queue.put(f"All extraction methods failed for: {url}")
    return None, metadata

def extract_forensic_insights(company: str, title: str, content: str, event_name: str) -> Dict:
    """
    Extract forensic insights from content using a two-stage approach
    """
    if not content or len(content.strip()) < 100:
        return None
        
    try:
        # Use AGENT_CONFIG for LLM initialization
        llm = ChatGoogleGenerativeAI(model=AGENT_CONFIG["model"], temperature=AGENT_CONFIG["temperature"])
        
        variables = {
            "company": company,
            "title": title,
            "event_name": event_name,
            "content": content
        }

        system_prompt, human_prompt  = prompt_manager.get_prompt("analyst_agent", "forensic_insights_extract", variables)
        
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = llm.invoke(messages)
        extracted_content = response.content.strip()

        if extracted_content == "NO_FORENSIC_CONTENT":
            progress_queue.put(f"No forensic content found in article: {title}")
            return None
        
        variables = {
            "company": company,
            "extracted_content": extracted_content
        }

        system_prompt, human_prompt = prompt_manager.get_prompt("analyst_agent", "forensic_insights_analysis", variables)
        
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        analysis_response = llm.invoke(messages)
        
        response_content = analysis_response.content.strip()
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content
            
        forensic_insights = json.loads(json_content)
        
        forensic_insights["raw_extract"] = extracted_content
        forensic_insights["article_title"] = title
        forensic_insights["event_category"] = event_name
        
        progress_queue.put(f"Successfully extracted forensic insights from: {title}")
        return forensic_insights
        
    except Exception as e:
        progress_queue.put(f"Error during content analysis: {str(e)[:100]}...")
        progress_queue.put(f"Traceback: {traceback.format_exc()[:200]}...")
        return None

def process_article_worker(args: Tuple) -> Optional[Dict]:
    """
    Worker function for processing a single article in a separate thread.
    The args tuple contains (company, event_name, article_info, article_index, total_articles)
    """
    company, event_name, article_info, article_index, total_articles = args
    
    article_title = article_info["title"]
    article_url = article_info["link"]
    
    try:
        progress_queue.put(f"[{article_index+1}/{total_articles}] Processing: {article_title}")
        
        content, metadata = fetch_article_content(article_url)
        
        processing_stats.increment("processed_articles")
        
        if not content:
            progress_queue.put(f"Failed to fetch content for: {article_url}")
            processing_stats.increment("failed_articles")
            return None
        
        insights = extract_forensic_insights(company, article_title, content, event_name)
        
        if not insights:
            progress_queue.put(f"No relevant forensic insights found in: {article_title}")
            return None
        
        processing_stats.increment("articles_with_insights")
        
        insights["url"] = article_url
        insights["metadata"] = metadata
        
        progress_queue.put(f"Successfully processed article: {article_title}")
        return insights
        
    except Exception as e:
        progress_queue.put(f"Error processing article {article_title}: {str(e)[:100]}...")
        progress_queue.put(f"Traceback: {traceback.format_exc()[:200]}...")
        processing_stats.increment("failed_articles")
        return None

def synthesize_event_insights(company: str, event_name: str, insights_list: List[Dict]) -> Dict:
    """
    Synthesize multiple insights about the same event to create a consolidated view.
    """
    if not insights_list or len(insights_list) == 0:
        return None
        
    progress_queue.put(f"Synthesizing {len(insights_list)} insights for event: {event_name}")  
    try:
        # Use AGENT_CONFIG for LLM initialization
        llm = ChatGoogleGenerativeAI(model=AGENT_CONFIG["model"], temperature=AGENT_CONFIG["temperature"])
        
        simplified_insights = []
        for insight in insights_list:
            simplified = {k: v for k, v in insight.items() if k not in ["raw_extract", "metadata"]}
            for key, value in simplified.items():
                if isinstance(value, str) and len(value) > 1000:
                    simplified[key] = value[:1000] + "... [truncated]"
            simplified_insights.append(simplified)
        
        variables = {
        "company": company,
        "event_name": event_name,
        "num_sources": len(simplified_insights),
        "insights": json.dumps(simplified_insights, indent=2)
        }
        system_prompt, human_prompt = prompt_manager.get_prompt("analyst_agent", "event_insight", variables) 
        
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
            
        synthesis = json.loads(json_content)
        
        progress_queue.put(f"Successfully synthesized insights for event: {event_name}")
        return synthesis
        
    except Exception as e:
        progress_queue.put(f"Error during synthesis for event {event_name}: {str(e)[:100]}...")
        progress_queue.put(f"Traceback: {traceback.format_exc()[:200]}...")
        
        return {
            "cross_validation": "Could not synthesize due to technical error",
            "timeline": [{"date": "Unknown", "description": "Event occurred"}],
            "key_entities": [{"name": company, "role": "Subject company"}],
            "evidence_assessment": "Error during synthesis",
            "severity_assessment": "Unknown",
            "credibility_score": 5,
            "red_flags": ["Technical error prevented complete analysis"],
            "narrative": f"Analysis of {event_name} involving {company} could not be completed due to technical error."
        }

def generate_company_analysis(company: str, events_synthesis: Dict, guidance: Dict = None) -> Dict:
    """
    Generate comprehensive company analysis based on all synthesized events.
    """
    progress_queue.put(f"Generating comprehensive analysis for {company} based on {len(events_synthesis)} events")
    
    try:
        # Use AGENT_CONFIG for LLM initialization
        llm = ChatGoogleGenerativeAI(model=AGENT_CONFIG["model"], temperature=AGENT_CONFIG["temperature"])
        
        simplified_events = {}
        for event_name, event_data in events_synthesis.items():
            event_copy = event_data.copy()
            if "narrative" in event_copy and len(event_copy["narrative"]) > 500:
                event_copy["narrative"] = event_copy["narrative"][:500] + "... [truncated]"
            simplified_events[event_name] = event_copy
        
        simplified_guidance = None
        if guidance:
            simplified_guidance = {k: v for k, v in guidance.items() if k in ["focus_areas", "priorities", "red_flags"]}
        
        variables = {
            "company": company,
            "num_events": len(simplified_events),
            "event_syntheses": json.dumps(simplified_events, indent=2),
            "guidance": json.dumps(simplified_guidance, indent=2) if simplified_guidance else "No specific guidance provided"
        }
        system_prompt, human_prompt = prompt_manager.get_prompt("analyst_agent", "event_insight", variables) 
        
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
            
        analysis = json.loads(json_content)
        
        progress_queue.put(f"Successfully generated comprehensive analysis for {company}")
        return analysis
        
    except Exception as e:
        progress_queue.put(f"Error during comprehensive analysis: {str(e)[:100]}...")
        progress_queue.put(f"Traceback: {traceback.format_exc()[:200]}...")
        
        return {
            "executive_summary": f"Analysis of {company} could not be completed due to technical error.",
            "risk_assessment": {
                "financial_integrity_risk": "Unknown",
                "legal_regulatory_risk": "Unknown",
                "reputational_risk": "Unknown",
                "operational_risk": "Unknown"
            },
            "key_patterns": ["Technical error prevented pattern analysis"],
            "critical_entities": [{"name": company, "role": "Subject company"}],
            "red_flags": ["Analysis incomplete due to technical error"],
            "timeline": [{"date": "Unknown", "description": "Analysis attempted"}],
            "forensic_assessment": "Analysis could not be completed",
            "report_markdown": f"# Forensic Analysis of {company}\n\nAnalysis could not be completed due to technical error."
        }

def progress_monitor():
    """
    Monitor and log progress information from the queue
    """
    while True:
        try:
            message = progress_queue.get(timeout=0.5)
            print(f"[Analyst Agent] {message}")
            progress_queue.task_done()
        except queue.Empty:
            if progress_queue.empty() and getattr(progress_monitor, "stop_flag", False):
                break
            continue
        except Exception as e:
            print(f"[Analyst Agent] Error in progress monitor: {e}")

def analyst_agent(state: Dict) -> Dict:
    """
    Enhanced analyst agent with parallel processing for efficient article analysis.
    """
    print("[Analyst Agent] Starting multithreaded analysis process...")
    
    company = state.get("company", "")
    research_results = state.get("research_results", {})
    analysis_guidance = state.get("analysis_guidance", {})
    
    if not company:
        print("[Analyst Agent] ERROR: Company name missing!")
        return {**state, "goto": "meta_agent_final", "analyst_status": "ERROR", "error": "Company name missing"}
    
    if not research_results:
        print("[Analyst Agent] ERROR: No research results to analyze!")
        return {**state, "goto": "meta_agent_final", "analyst_status": "ERROR", "error": "No research results"}
    
    print(f"[Analyst Agent] Analyzing {len(research_results)} events for company: {company}")
    
    processing_stats.total_events = len(research_results)
    processing_stats.total_articles = sum(len(articles) for articles in research_results.values())
    
    analysis_results = {
        "forensic_insights": {},    
        "event_synthesis": {},      
        "company_analysis": {},     
        "red_flags": [],            
        "evidence_map": {},         
        "entity_network": {},       
        "timeline": [],             
    }
    
    import threading
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        all_event_futures = {}
        
        for event_name, articles in research_results.items():
            print(f"[Analyst Agent] Submitting {len(articles)} articles for event: {event_name}")
            
            article_tasks = [
                (company, event_name, article, i, len(articles)) 
                for i, article in enumerate(articles)
            ]
            
            futures = [
                executor.submit(process_article_worker, task_args) 
                for task_args in article_tasks
            ]
            
            all_event_futures[event_name] = futures
        
        for event_name, futures in all_event_futures.items():
            event_insights = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        event_insights.append(result)
                except Exception as e:
                    print(f"[Analyst Agent] Error in article processing task: {e}")
                    processing_stats.increment("failed_articles")
            
            if event_insights:
                print(f"[Analyst Agent] Collected {len(event_insights)} insights for event: {event_name}")
                analysis_results["forensic_insights"][event_name] = event_insights
                knowledge_base.store_event_insights(event_name, event_insights)
                processing_stats.increment("events_with_insights")
                
                event_synthesis = synthesize_event_insights(company, event_name, event_insights)
                if event_synthesis:
                    analysis_results["event_synthesis"][event_name] = event_synthesis
                    
                    if "timeline" in event_synthesis:
                        timeline_items = []
                        for timeline_item in event_synthesis.get("timeline", []):
                            if "date" in timeline_item and timeline_item["date"] != "Unknown":
                                timeline_item["event"] = event_name
                                timeline_items.append(timeline_item)
                        if timeline_items:
                            knowledge_base.add_to_timeline(timeline_items)
                            analysis_results["timeline"].extend(timeline_items)
                    
                    if "red_flags" in event_synthesis:
                        knowledge_base.add_red_flags(event_synthesis.get("red_flags", []))
                        for flag in event_synthesis.get("red_flags", []):
                            if flag not in analysis_results["red_flags"]:
                                analysis_results["red_flags"].append(flag)
                    
                    if "key_entities" in event_synthesis:
                        entities = {}
                        for entity_info in event_synthesis.get("key_entities", []):
                            if "name" in entity_info and entity_info["name"] != "Unknown":
                                entity_name = entity_info["name"]
                                entities[entity_name] = entity_info
                        if entities:
                            knowledge_base.update_entities(entities)
            else:
                print(f"[Analyst Agent] No insights collected for event: {event_name}")
    
    progress_monitor.stop_flag = True
    monitor_thread.join(timeout=2)
    
    analysis_results["timeline"] = sorted(
        analysis_results["timeline"], 
        key=lambda x: datetime.fromisoformat(x["date"]) if re.match(r'\d{4}-\d{2}-\d{2}', x.get("date", "")) else datetime.now(),
        reverse=True
    )
    
    if analysis_results["event_synthesis"]:
        company_analysis = generate_company_analysis(
            company, 
            analysis_results["event_synthesis"],
            analysis_guidance
        )
        analysis_results["company_analysis"] = company_analysis
        
        final_report = company_analysis.get("report_markdown", f"# Forensic Analysis of {company}\n\nNo significant findings.")
    else:
        print(f"[Analyst Agent] No significant forensic insights found for {company}")
        stats = processing_stats.get_stats()
        final_report = f"""
        # Forensic Analysis of {company}
        
        ## Executive Summary
        
        After analyzing {stats['total_articles']} articles across {stats['total_events']} potential events, 
        no significant forensic concerns were identified for {company}. The available information does not 
        indicate material issues related to financial integrity, regulatory compliance, or corporate governance.
        
        ## Analysis Process
        
        - Total events examined: {stats['total_events']}
        - Total articles processed: {stats['processed_articles']}
        - Articles with potential forensic content: {stats['articles_with_insights']}
        - Events with synthesized insights: {stats['events_with_insights']}
        
        ## Conclusion
        
        Based on the available information, there are no significant red flags or forensic concerns 
        that would warrant further investigation at this time.
        """
    
    state["analysis_results"] = analysis_results
    state["final_report"] = final_report
    state["analyst_status"] = "DONE"
    state["analysis_stats"] = processing_stats.get_stats()
    
    print(f"[Analyst Agent] Analysis complete.")
    print(f"[Analyst Agent] Processed {processing_stats.processed_articles}/{processing_stats.total_articles} articles.")
    print(f"[Analyst Agent] Found forensic insights in {processing_stats.articles_with_insights} articles.")
    print(f"[Analyst Agent] Failed to process {processing_stats.failed_articles} articles.")
    
    return {**state, "goto": "meta_agent_final"}