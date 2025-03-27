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
    Multi-Agent System for forensic analysis of news content.
    Orchestrates a workflow of specialized agents to research, analyze,
    and report on potential issues related to a target company.
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
        Build the agent workflow graph with the enhanced agents.
        """
        logger.info("Building the agent workflow graph")
        
        graph = StateGraph(dict)
        
        graph.add_node("meta_agent", meta_agent)
        graph.add_node("research_agent", research_agent)
        graph.add_node("analyst_agent", analyst_agent)
        graph.add_node("meta_agent_final", meta_agent_final)
        
        if self.config.get("enable_error_handling", True):
            graph.add_node("error_handler", self._error_handler)
        
        graph.add_edge(START, "meta_agent")
        
        graph.add_conditional_edges(
            "meta_agent",
            self._meta_agent_router
        )
        
        graph.add_edge("research_agent", "meta_agent")
        graph.add_edge("analyst_agent", "meta_agent_final")
        
        self.graph = graph
        return graph
    
    def _meta_agent_router(self, state: Dict) -> str:
        """
        Enhanced routing logic for the meta agent.
        Makes decisions based on research quality and analysis guidance.
        """
        try:
            if state.get("goto") in ["research_agent", "analyst_agent"]:
                return state["goto"]
            
            quality_assessment = state.get("quality_assessment", {})
            overall_score = quality_assessment.get("overall_score", 0)
            
            research_results = state.get("research_results", {})
            num_events = len(research_results)
            
            if num_events < 3 or overall_score < 6:
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
        
        if "research_agent" in str(error):
            logger.info("Attempting recovery: Skipping to analyst with available data")
            return {**state, "goto": "analyst_agent", "recovery_applied": True}
        elif "analyst_agent" in str(error):
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
    
    def run(self, company_name: str, industry: str = None, max_iterations: int = 6) -> Dict:
        """
        Run the full News Forensic workflow for a given company.
        
        Args:
            company_name: The name of the company to analyze
            industry: Optional industry context for better research
            max_iterations: Maximum number of iterations to prevent infinite loops
                
        Returns:
            The final state containing analysis results and final report
        """
        logger.info(f"Starting analysis for company: {company_name}")
        
        initial_state = {
            "company": company_name,
            "industry": industry,
            "research_plan": [],
            "domains_explored": [],
            "research_results": {},
            "analysis_results": {},
            "analyst_status": "",
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
            config={"configurable": {"thread_id": f"{company_name}_{self.start_time.strftime('%Y%m%d%H%M%S')}"}}
        )
        
        iteration = 0
        while current_state.get("goto") != END and iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}: Current state goto={current_state.get('goto')}")
            
            current_state["iterations"] = iteration
            
            # Only save snapshots occasionally to improve performance
            if iteration % 3 == 0 or current_state.get("goto") in ["analyst_agent", "meta_agent_final"]:
                self._save_state_snapshot(current_state, f"iter_{iteration}")
            
            # Add minimal delay only when necessary
            if current_state.get("goto") == "research_agent":
                time.sleep(0.5)
            
            # Accelerate transitions based on quality thresholds
            if current_state.get("quality_assessment", {}).get("overall_score", 0) >= 6 and iteration > 1:
                if current_state.get("goto") == "research_agent":
                    logger.info("Quality threshold met, accelerating to analysis phase")
                    current_state["goto"] = "analyst_agent"
            
            # Skip additional research iterations if we have enough data
            if iteration >= 3 and current_state.get("research_results") and len(current_state.get("research_results", {})) > 3:
                if current_state.get("goto") == "research_agent" and not current_state.get("analysis_results"):
                    logger.info("Sufficient research data collected, proceeding to analysis")
                    current_state["goto"] = "analyst_agent"
            
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
        logger.info(f"Analysis completed for {company_name} after {iteration} iterations")
        
        return current_state
    
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
        
        return {
            "company": self.final_state.get("company"),
            "runtime": str(datetime.now() - self.start_time),
            "iterations": self.final_state.get("iterations", 0),
            "events_analyzed": len(self.final_state.get("research_results", {})),
            "status": "Success" if self.final_state.get("final_report") else "Failed",
            "error": self.final_state.get("error"),
            "warning": self.final_state.get("warning"),
        }