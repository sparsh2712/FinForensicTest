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

from backend.agents.rag_agent import rag_agent
from backend.agents.youtube_agent import youtube_agent
from backend.agents.corporate_agent import corporate_agent
from backend.agents.corporate_meta_writer_agent import corporate_meta_writer_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("corporate_governance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("corporate_governance")

class CorporateGovernanceSystem:
    """
    Multi-Agent System for corporate governance analysis.
    Orchestrates a workflow of specialized agents to analyze corporate documents, 
    financial videos, and corporate filings to generate a comprehensive corporate governance report.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Corporate Governance System with configuration options.
        """
        self.config = config or {}
        self.start_time = datetime.now()
        self.graph = None
        self.app = None
        self.final_state = None
        
        os.makedirs("corporate_reports", exist_ok=True)
        os.makedirs("debug", exist_ok=True)
        os.makedirs("debug/corporate_reports", exist_ok=True)
        os.makedirs("debug/state_snapshots", exist_ok=True)
        
        logger.info("CorporateGovernanceSystem initialized with config: %s", self.config)
        
    def build_graph(self) -> StateGraph:
        """
        Build the agent workflow graph for corporate governance analysis.
        """
        logger.info("Building the agent workflow graph for corporate governance")
        
        graph = StateGraph(dict)
        
        # Add nodes for each agent
        graph.add_node("rag_agent", rag_agent)
        graph.add_node("youtube_agent", youtube_agent)
        graph.add_node("corporate_agent", corporate_agent)
        graph.add_node("corporate_meta_writer_agent", corporate_meta_writer_agent)
        
        if self.config.get("enable_error_handling", True):
            graph.add_node("error_handler", self._error_handler)
        
        # Define the workflow starting point
        graph.add_edge(START, "corporate_meta_writer_agent")
        
        # Add conditional edges for routing between agents
        graph.add_conditional_edges(
            "corporate_meta_writer_agent",
            self._corporate_meta_writer_router
        )
        
        # Connect other agent outputs back to corporate_meta_writer_agent
        graph.add_edge("rag_agent", "corporate_meta_writer_agent")
        graph.add_edge("youtube_agent", "corporate_meta_writer_agent")
        graph.add_edge("corporate_agent", "corporate_meta_writer_agent")
        
        self.graph = graph
        return graph
    
    def _corporate_meta_writer_router(self, state: Dict) -> str:
        """
        Routing logic for the corporate_meta_writer_agent.
        Routes to the appropriate agent based on the 'goto' field in the state.
        """
        try:
            if state.get("goto") == "rag_agent":
                logger.info("Routing to RAG Agent for document analysis")
                return "rag_agent"
            elif state.get("goto") == "youtube_agent":
                logger.info(f"Routing to YouTube Agent for {state.get('youtube_agent_action', 'unknown')} action")
                return "youtube_agent"
            elif state.get("goto") == "corporate_agent":
                logger.info("Routing to Corporate Agent for company data collection")
                return "corporate_agent"
            elif state.get("goto") == "END":
                logger.info("Workflow complete, ending process")
                return END
            else:
                logger.warning(f"Unknown goto value: {state.get('goto')}. Defaulting to corporate_meta_writer_agent.")
                return "corporate_meta_writer_agent"
        except Exception as e:
            logger.error(f"Error in corporate_meta_writer_router: {e}")
            return "error_handler"
    
    def _error_handler(self, state: Dict) -> Dict:
        """
        Handle errors in the workflow and attempt recovery.
        """
        error = state.get("error", "Unknown error")
        logger.error(f"Error handler activated: {error}")
        
        self._save_state_snapshot(state, "error")
        
        if "rag_agent" in str(error):
            logger.info("Attempting recovery: Skipping RAG analysis and continuing workflow")
            state["rag_status"] = "ERROR"
            state["goto"] = "corporate_meta_writer_agent"
            state["corporate_meta_step"] = "post_rag"
            return state
        elif "youtube_agent" in str(error):
            logger.info("Attempting recovery: Skipping YouTube analysis and continuing workflow")
            state["youtube_status"] = "ERROR"
            if state.get("youtube_agent_action") == "search":
                state["goto"] = "corporate_meta_writer_agent"
                state["corporate_meta_step"] = "select_videos"
                state["search_results"] = {}
            else:  # transcribe action
                state["goto"] = "corporate_meta_writer_agent"
                state["corporate_meta_step"] = "post_transcription"
                state["transcript_results"] = []
            return state
        elif "corporate_agent" in str(error):
            logger.info("Attempting recovery: Skipping corporate data collection and generating report with available data")
            state["corporate_status"] = "ERROR"
            state["goto"] = "corporate_meta_writer_agent"
            state["corporate_meta_step"] = "generate_report"
            state["corporate_results"] = {}
            return state
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
            filename = f"debug/state_snapshots/{company}_cg_{marker}_{timestamp}.json"
            
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
        output_filename = os.path.join("corporate_reports", f"{company_name.replace(' ', '_')}_{timestamp}.md")
        
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Final report saved to {output_filename}")
            
            latest_filename = os.path.join("corporate_reports", f"{company_name.replace(' ', '_')}_latest.md")
            with open(latest_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            
        except Exception as e:
            logger.error(f"Error saving final report: {e}")
    
    def run(self, company_name: str, company_symbol: str, pdf_path: str, max_iterations: int = 20) -> Dict:
        """
        Run the full Corporate Governance workflow for a given company.
        
        Args:
            company_name: The name of the company to analyze
            company_symbol: The stock symbol of the company
            pdf_path: Path to the company report PDF
            max_iterations: Maximum number of iterations to prevent infinite loops
            
        Returns:
            The final state containing analysis results and final report
        """
        logger.info(f"Starting corporate governance analysis for company: {company_name} ({company_symbol})")
        
        initial_state = {
            "company": company_name,
            "company_symbol": company_symbol,
            "pdf_path": pdf_path,
            "is_file_embedded": False,
            "corporate_meta_step": "start",
            "final_report": "",
            "report_sections": [],
            "start_time": self.start_time.isoformat(),
            "iterations": 0
        }
        
        if not self.graph:
            self.build_graph()
        
        checkpoint_saver = MemorySaver()
        self.app = self.graph.compile(checkpointer=checkpoint_saver)
        
        self._save_state_snapshot(initial_state, "initial")
        
        logger.info("Invoking initial state")
        current_state = self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"{company_name}_cg_{self.start_time.strftime('%Y%m%d%H%M%S')}"}}
        )
        
        iteration = 0
        while current_state.get("goto") != END and iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}: Current state goto={current_state.get('goto')}")
            
            current_state["iterations"] = iteration
            
            if iteration % 5 == 0 or current_state.get("goto") == END:
                self._save_state_snapshot(current_state, f"iter_{iteration}")
            
            time.sleep(1)
            
            current_state = self.app.invoke(
                current_state,
                config={"configurable": {"thread_id": f"{company_name}_cg_{self.start_time.strftime('%Y%m%d%H%M%S')}"}}
            )
        
        if iteration >= max_iterations and current_state.get("goto") != END:
            logger.warning(f"Reached maximum iterations ({max_iterations}) without completion")
            current_state["warning"] = f"Analysis terminated after reaching maximum iterations ({max_iterations})"
            current_state["goto"] = END
        
        self._save_state_snapshot(current_state, "final")
        
        self._save_final_report(current_state)
        
        self.final_state = current_state
        logger.info(f"Corporate governance analysis completed for {company_name} after {iteration} iterations")
        
        return current_state
    
    def get_summary(self) -> Dict:
        """
        Return a summary of the analysis process.
        """
        if not self.final_state:
            return {"status": "Not run yet"}
        
        return {
            "company": self.final_state.get("company"),
            "company_symbol": self.final_state.get("company_symbol"),
            "runtime": str(datetime.now() - self.start_time),
            "iterations": self.final_state.get("iterations", 0),
            "status": "Success" if self.final_state.get("final_report") else "Failed",
            "error": self.final_state.get("error"),
            "warning": self.final_state.get("warning"),
        }