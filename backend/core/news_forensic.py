import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Callable
import traceback

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.agents.meta_agent import meta_agent
from backend.agents.research_agent import research_agent
from backend.agents.analyst_agent import analyst_agent
from backend.agents.meta_agent_final import meta_agent_final

# Import our new agents
from backend.agents.corporate_agent import corporate_agent
from backend.agents.youtube_agent import youtube_agent
from backend.agents.rag_agent import rag_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_forensic.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("news_forensic")

class NewsForensicSystem:
    """
    Enhanced Multi-Agent System for forensic analysis of news content.
    Now includes Corporate, YouTube, and RAG agents for comprehensive analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the News Forensic System with configuration options.
        """
        self.config = config or {}
        self.start_time = datetime.now()
        self.graph = None
        self.app = None
        self.final_state = None
        
        os.makedirs("markdowns", exist_ok=True)
        os.makedirs("debug", exist_ok=True)
        os.makedirs("debug/reports", exist_ok=True)
        os.makedirs("debug/state_snapshots", exist_ok=True)
        
        logger.info("NewsForensicSystem initialized with config: %s", self.config)
    
    def build_graph(self) -> StateGraph:
        """
        Build the enhanced agent workflow graph with additional agents.
        """
        logger.info("Building the enhanced agent workflow graph")
        
        graph = StateGraph(dict)
        
        # Add all agents to the graph
        graph.add_node("meta_agent", meta_agent)
        graph.add_node("research_agent", research_agent)
        graph.add_node("analyst_agent", analyst_agent)
        graph.add_node("meta_agent_final", meta_agent_final)
        
        # Add new agents
        graph.add_node("corporate_agent", corporate_agent)
        graph.add_node("youtube_agent", youtube_agent)
        graph.add_node("rag_agent", rag_agent)
        
        if self.config.get("enable_error_handling", True):
            graph.add_node("error_handler", self._error_handler)
        
        graph.add_edge(START, "meta_agent")
        
        # Add conditional routing from meta_agent
        graph.add_conditional_edges(
            "meta_agent",
            self._meta_agent_router
        )
        
        # Add edges from research_agent and our new agents back to meta_agent
        graph.add_edge("research_agent", "meta_agent")
        graph.add_edge("corporate_agent", "meta_agent")
        graph.add_edge("youtube_agent", "meta_agent")
        graph.add_edge("rag_agent", "meta_agent")
        
        # Add edge from analyst to meta_agent_final
        graph.add_edge("analyst_agent", "meta_agent_final")
        
        self.graph = graph
        return graph
    
    def _meta_agent_router(self, state: Dict) -> str:
        """
        Enhanced routing logic for the meta agent.
        Now handles routing to the new agents as well.
        """
        try:
            # First check if an explicit "goto" is set
            if state.get("goto") in ["research_agent", "analyst_agent", "corporate_agent", "youtube_agent", "rag_agent"]:
                logger.info(f"Router: Explicit route to {state['goto']}")
                return state["goto"]
            
            # Check if we should run corporate agent
            if state.get("run_corporate_agent", False) and state.get("corporate_status") != "DONE":
                logger.info("Router: Directing to corporate_agent")
                return "corporate_agent"
            
            # Check if we should run YouTube agent
            if state.get("run_youtube_agent", False) and state.get("youtube_status") != "DONE":
                logger.info("Router: Directing to youtube_agent")
                return "youtube_agent"
            
            # Check if we should run RAG agent
            if state.get("run_rag_agent", False) and state.get("rag_status") != "DONE":
                logger.info("Router: Directing to rag_agent")
                return "rag_agent"
            
            # Otherwise, use the research quality to decide between research and analyst
            quality_assessment = state.get("quality_assessment", {})
            overall_score = quality_assessment.get("overall_score", 0)
            
            research_results = state.get("research_results", {})
            num_events = len(research_results)
            
            # If all additional agents are done and we have sufficient research,
            # proceed to analyst, otherwise continue with research
            if (num_events < 3 or overall_score < 6) and not state.get("force_analysis", False):
                logger.info(f"Router: Directing to research_agent (events: {num_events}, quality: {overall_score})")
                return "research_agent"
            else:
                logger.info(f"Router: Directing to analyst_agent (events: {num_events}, quality: {overall_score})")
                return "analyst_agent"
        except Exception as e:
            logger.error(f"Error in meta_agent_router: {e}")
            return "research_agent"
    
    def _error_handler(self, state: Dict) -> Dict:
        """
        Handle errors in the workflow and attempt recovery.
        """
        error = state.get("error", "Unknown error")
        logger.error(f"Error handler activated: {error}")
        
        self._save_state_snapshot(state, "error")
        
        # Identify which agent caused the error
        error_agent = None
        for agent in ["research_agent", "corporate_agent", "youtube_agent", "rag_agent", "analyst_agent"]:
            if agent in str(error):
                error_agent = agent
                break
        
        # Apply recovery strategies based on the agent that failed
        if error_agent in ["research_agent", "corporate_agent", "youtube_agent", "rag_agent"]:
            # For data collection agents, mark them as completed with error and continue
            logger.info(f"Attempting recovery: Marking {error_agent} as complete with error")
            state[f"{error_agent.split('_')[0]}_status"] = "ERROR"
            
            # Force progress to analysis if we have enough data
            if state.get("research_results") and any([
                state.get("corporate_results"),
                state.get("youtube_results"),
                state.get("rag_results")
            ]):
                logger.info("Some data available. Proceeding to analysis.")
                state["force_analysis"] = True
                return {**state, "goto": "meta_agent", "recovery_applied": True}
            else:
                # Not enough data, try to get basic research
                logger.info("Insufficient data. Redirecting to research agent.")
                return {**state, "goto": "research_agent", "recovery_applied": True}
                
        elif error_agent == "analyst_agent":
            logger.info("Attempting recovery: Proceeding to final report with limited analysis")
            return {**state, "goto": "meta_agent_final", "analyst_status": "ERROR", "recovery_applied": True}
        else:
            logger.info("Cannot recover. Terminating workflow.")
            return {**state, "goto": END, "error": error, "recovery_applied": False}
    
    def _save_state_snapshot(self, state: Dict, marker: str = ""):
        """
        Save a snapshot of the current state for debugging purposes.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company = state.get("company", "unknown").replace(" ", "_")
            filename = f"debug/state_snapshots/{company}_{marker}_{timestamp}.json"
            
            debug_state = {}
            for key, value in state.items():
                if isinstance(value, dict) and len(str(value)) > 1000:
                    debug_state[key] = f"<Dict with {len(value)} items>"
                elif isinstance(value, str) and len(value) > 1000:
                    debug_state[key] = value[:1000] + "... [truncated]"
                else:
                    debug_state[key] = value
            
            with open(filename, "w") as f:
                json.dump(debug_state, f, indent=2, default=str)
            
            logger.debug(f"State snapshot saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save state snapshot: {e}")
    
    def run(self, company_name: str, industry: str = None, website: str = None, 
            max_iterations: int = 20, run_corporate: bool = False, run_youtube: bool = False, 
            run_rag: bool = False, rag_pdf_path: str = None, youtube_queries: List[str] = None,
            rag_queries: List[str] = None, corporate_streams: List[str] = None) -> Dict:
        """
        Run the enhanced News Forensic workflow for a given company.
        
        Args:
            company_name: The name of the company to analyze
            industry: Optional industry context for better research
            website: Optional company website
            max_iterations: Maximum number of iterations to prevent infinite loops
            run_corporate: Whether to run the corporate agent
            run_youtube: Whether to run the YouTube agent
            run_rag: Whether to run the RAG agent
            rag_pdf_path: Path to PDF for RAG analysis
            youtube_queries: Custom YouTube search queries
            rag_queries: Custom RAG queries
            corporate_streams: List of corporate data streams to fetch
            
        Returns:
            The final state containing analysis results and final report
        """
        logger.info(f"Starting enhanced analysis for company: {company_name}")
        
        # Build initial state with options for all agents
        initial_state = {
            "company": company_name,
            "industry": industry,
            "company_website": website,
            "research_plan": [],
            "domains_explored": [],
            "research_results": {},
            "analysis_results": {},
            "analyst_status": "",
            "final_report": "",
            "report_sections": [],
            "start_time": self.start_time.isoformat(),
            "iterations": 0,
            
            # Corporate agent settings
            "run_corporate_agent": run_corporate,
            "corporate_status": "",
            "corporate_stream_config": self._prepare_corporate_streams(corporate_streams) if run_corporate else None,
            
            # YouTube agent settings
            "run_youtube_agent": run_youtube,
            "youtube_status": "",
            "youtube_queries": youtube_queries,
            
            # RAG agent settings
            "run_rag_agent": run_rag and rag_pdf_path is not None,
            "rag_status": "",
            "rag_pdf_path": rag_pdf_path,
            "rag_queries": rag_queries
        }
        
        if not self.graph:
            self.build_graph()
        
        checkpoint_saver = MemorySaver()
        self.app = self.graph.compile(checkpointer=checkpoint_saver)
        
        self._save_state_snapshot(initial_state, "initial")
        
        logger.info("Invoking initial state with enhanced configuration")
        current_state = self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"{company_name}_{self.start_time.strftime('%Y%m%d%H%M%S')}"}}
        )
        
        iteration = 0
        while current_state.get("goto") != END and iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}: Current state goto={current_state.get('goto')}")
            
            current_state["iterations"] = iteration
            
            if iteration % 5 == 0 or current_state.get("goto") in ["analyst_agent", "meta_agent_final"]:
                self._save_state_snapshot(current_state, f"iter_{iteration}")
            
            time.sleep(1)
            
            current_state = self.app.invoke(
                current_state,
                config={"configurable": {"thread_id": f"{company_name}_{self.start_time.strftime('%Y%m%d%H%M%S')}"}}
            )
        
        if iteration >= max_iterations and current_state.get("goto") != END:
            logger.warning(f"Reached maximum iterations ({max_iterations}) without completion")
            current_state["warning"] = f"Analysis terminated after reaching maximum iterations ({max_iterations})"
            current_state["goto"] = END
        
        self._save_state_snapshot(current_state, "final")
        
        self._save_final_report(current_state)
        
        self.final_state = current_state
        logger.info(f"Enhanced analysis completed for {company_name} after {iteration} iterations")
        
        return current_state
    
    def _prepare_corporate_streams(self, streams: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Prepare the corporate stream configuration based on user selection"""
        all_streams = {
            "BoardMeetings": {
                "active": False,
                "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
            },
            "Announcements": {
                "active": False,
                "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
            },
            "CorporateActions": {
                "active": False,
                "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
            },
            "AnnualReports": {
                "active": False,
                "input_params": {}
            },
            "FinancialResults": {
                "active": False,
                "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
            }
        }
        
        # If no streams specified, activate them all
        if not streams:
            for stream in all_streams:
                all_streams[stream]["active"] = True
            return all_streams
        
        # Otherwise, activate only the specified streams
        for stream in streams:
            if stream in all_streams:
                all_streams[stream]["active"] = True
        
        return all_streams
    
    def _save_final_report(self, state: Dict) -> None:
        """
        Save the final report to the appropriate location.
        """
        company_name = state.get("company", "unknown_company")
        report_content = state.get("final_report", "")
        
        if not report_content:
            logger.error("No report content generated!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join("markdowns", f"{company_name.replace(' ', '_')}_{timestamp}.md")
        
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Final report saved to {output_filename}")
            
            latest_filename = os.path.join("markdowns", f"{company_name.replace(' ', '_')}_latest.md")
            with open(latest_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            
        except Exception as e:
            logger.error(f"Error saving final report: {e}")
    
    def get_summary(self) -> Dict:
        """
        Return a summary of the analysis process.
        """
        if not self.final_state:
            return {"status": "Not run yet"}
        
        agents_run = []
        for agent in ["research", "corporate", "youtube", "rag", "analyst"]:
            if self.final_state.get(f"{agent}_status") == "DONE":
                agents_run.append(f"{agent}_agent")
        
        return {
            "company": self.final_state.get("company"),
            "runtime": str(datetime.now() - self.start_time),
            "iterations": self.final_state.get("iterations", 0),
            "events_analyzed": len(self.final_state.get("research_results", {})),
            "agents_run": agents_run,
            "status": "Success" if self.final_state.get("final_report") else "Failed",
            "error": self.final_state.get("error"),
            "warning": self.final_state.get("warning"),
        }