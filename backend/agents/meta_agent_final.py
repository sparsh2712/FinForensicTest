import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END
from dotenv import load_dotenv
load_dotenv()
from backend.utils.prompt_manager import PromptManager

prompt_manager = PromptManager("/home/satyam/Desktop/FinForensicTest/backend/prompts")

def select_top_events(events: Dict, event_metadata: Dict, max_detailed_events: int = 6) -> Tuple[List[str], List[str]]:
    """
    Select the top events for detailed coverage and the rest for the "Other Notable Events" section.
    
    Args:
        events: Dictionary of events and their articles
        event_metadata: Dictionary of event metadata including importance scores
        max_detailed_events: Maximum number of events to cover in detail
        
        
    Returns:
        Tuple of (top_event_names, other_event_names)
    """
    sorted_events = sorted(
        [(name, event_metadata.get(name, {}).get("importance_score", 0)) 
         for name in events.keys()],
        key=lambda x: x[1],
        reverse=True
    )
    
    top_events = [name for name, _ in sorted_events[:max_detailed_events]]
    other_events = [name for name, _ in sorted_events[max_detailed_events:]]
    
    return top_events, other_events

def generate_detailed_event_section(company: str, event_name: str, event_data: List[Dict], llm) -> str:
    """
    Generate a detailed analysis section for a high-priority event.
    """
    print(f"[Meta Agent Final] Generating detailed analysis for event: {event_name}")

    articles = json.dumps([{
        "title": a.get("title", ""),
        "source": a.get("source", "Unknown"),
        "date": a.get("date", "Unknown"),
        "snippet": a.get("snippet", "")
    } for a in event_data], indent=2)

    variables = {
        "event_name": event_name,
        "company": company,
        "articles": articles
    }
    system_prompt, human_prompt = prompt_manager.get_prompt("meta_agent_final", "detailed_event_section", variables)
    messages = [
        ("system", system_prompt),
        ("human", human_prompt)
    ]
    try:
        response = llm.invoke(messages)
        detailed_section = response.content.strip()
        return f"## {event_name}\n\n{detailed_section}\n\n"
    except Exception as e:
        print(f"[Meta Agent Final] Error generating detailed event section: {e}")
        return f"## {event_name}\n\nUnable to generate detailed analysis due to technical error.\n\n"

def generate_other_events_section(company: str, events: Dict, event_metadata: Dict, other_event_names: List[str], llm) -> str:
    """
    Generate a summarized section covering other notable events.
    """
    if not other_event_names:
        return ""
        
    print(f"[Meta Agent Final] Generating summary for {len(other_event_names)} other events")
    
    event_summaries = []
    for event_name in other_event_names:
        articles = events.get(event_name, [])
        importance = event_metadata.get(event_name, {}).get("importance_score", 0)
        
        article_summaries = [{
            "title": a.get("title", ""),
            "source": a.get("source", "Unknown"),
            "date": a.get("date", "Unknown")
        } for a in articles[:3]]  
        
        event_summaries.append({
            "name": event_name,
            "importance": importance,
            "article_count": len(articles),
            "articles": article_summaries
        })
    
    variable = {
        "company": company,
        "events": json.dumps(event_summaries, indent=2)
    }
    system_prompt, human_prompt = prompt_manager.get_prompt("meta_agent_final", "other_event_section", variable)
    messages = [
        ("system", system_prompt),
        ("human", human_prompt)
    ]
    try:
        response = llm.invoke(messages)
        other_events_section = response.content.strip()
        return f"# Other Notable Events\n\n{other_events_section}\n\n"
    except Exception as e:
        print(f"[Meta Agent Final] Error generating other events section: {e}")
        return "# Other Notable Events\n\nUnable to generate summary of other events due to technical error.\n\n"

def generate_executive_summary(company: str, top_events: List[str], all_events: Dict, event_metadata: Dict, llm) -> str:
    """
    Generate an executive summary focusing on the most significant findings.
    """
    print(f"[Meta Agent Final] Generating executive summary with focus on top {len(top_events)} events")
    
    top_event_info = []
    for event_name in top_events:
        metadata = event_metadata.get(event_name, {})
        top_event_info.append({
            "name": event_name,
            "importance_score": metadata.get("importance_score", 0),
            "is_quarterly_report": metadata.get("is_quarterly_report", False)
        })

    variable = {
        "company": company,
        "top_event_info": json.dumps(top_event_info, indent=2),
        "total_events": len(all_events)
    }
    system_prompt, human_prompt = prompt_manager.get_prompt("meta_agent_final", "executive_summary", variable)
    messages = [
        ("system", system_prompt),
        ("human", human_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        summary = response.content.strip()
        return f"# Executive Summary\n\n{summary}\n\n"
    except Exception as e:
        print(f"[Meta Agent Final] Error generating executive summary: {e}")
        return "# Executive Summary\n\nUnable to generate executive summary due to technical error.\n\n"

def meta_agent_final(state: Dict) -> Dict:
    """
    Enhanced Meta Agent Final that generates a report with:
    - Detailed analysis of 5-6 most important events
    - Summarized section for other notable events
    - Focus on legal cases against the company and potential misconduct
    """
    print("[Meta Agent Final] Received state:", state)
    
    if state.get("analyst_status") != "DONE":
        print("[Meta Agent Final] Analyst not done yet. Waiting...")
        time.sleep(1)
        return {**state, "goto": "meta_agent_final"}
    
    print("[Meta Agent Final] Analyst work complete. Starting report generation.")
    
    company = state.get("company", "Unknown Company")
    research_results = state.get("research_results", {})
    event_metadata = state.get("event_metadata", {})
    analysis_results = state.get("analysis_results", {})
    
    report_sections = []
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        print("[Meta Agent Final] Initialized language model.")
        
        top_events, other_events = select_top_events(research_results, event_metadata, max_detailed_events=6)
        print(f"[Meta Agent Final] Selected {len(top_events)} events for detailed analysis and {len(other_events)} for summary")
        
        timestamp = datetime.now().strftime("%Y-%m-%d")
        report_sections.append(f"# Forensic News Analysis Report: {company}\n\nReport Date: {timestamp}\n\n")
        
        executive_summary = generate_executive_summary(company, top_events, research_results, event_metadata, llm)
        report_sections.append(executive_summary)
        
        detailed_events_section = "# Key Events Analysis\n\n"
        for event_name in top_events:
            event_data = research_results.get(event_name, [])
            if event_data:
                event_section = generate_detailed_event_section(company, event_name, event_data, llm)
                detailed_events_section += event_section
        
        report_sections.append(detailed_events_section)
        
        if other_events:
            other_events_section = generate_other_events_section(
                company, research_results, event_metadata, other_events, llm
            )
            report_sections.append(other_events_section)
        
        if len(top_events) > 1:
            events = json.dumps([{
                "name": event,
                "importance": event_metadata.get(event, {}).get("importance_score", 0)
            } for event in top_events], indent=2)

            variables = {
                "company": company,
                "events": events
            }
            system_prompt, human_prompt = prompt_manager.get_prompt("meta_agent_final", "pattern_analysis", variables)
            messages = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            try:
                response = llm.invoke(messages)
                pattern_section = response.content.strip()
                report_sections.append(f"# Pattern Recognition\n\n{pattern_section}\n\n")
            except Exception as e:
                print(f"[Meta Agent Final] Error generating pattern section: {e}")
        
        variables = {
            "company": company,
            "top_events": json.dumps([event for event in top_events], indent=2)
        }
        system_prompt, human_prompt = prompt_manager.get_prompt("meta_agent_final", "recommendation", variables)
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        try:
            response = llm.invoke(messages)
            recommendations = response.content.strip()
            report_sections.append(f"# Recommendations\n\n{recommendations}\n\n")
        except Exception as e:
            print(f"[Meta Agent Final] Error generating recommendations: {e}")
        
        full_report = "\n".join(report_sections)
        
        state["final_report"] = full_report
        state["report_sections"] = report_sections
        state["top_events"] = top_events
        state["other_events"] = other_events
        
        print("[Meta Agent Final] Report generation successfully completed.")
        
        try:
            debug_dir = "debug/reports"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
            
            debug_filename = f"{debug_dir}/{company.replace(' ', '_')}_{timestamp}.md"
            with open(debug_filename, "w") as f:
                f.write(full_report)
            print(f"[Meta Agent Final] Debug copy saved to {debug_filename}")
        except Exception as e:
            print(f"[Meta Agent Final] Could not save debug copy: {e}")
            
    except Exception as e:
        print(f"[Meta Agent Final] Error in report generation: {e}")
        
        fallback_report = f"""
        # Forensic News Analysis Report: {company}
        
        Report Date: {datetime.now().strftime("%Y-%m-%d")}
        
        ## Executive Summary
        
        This report presents the findings of a forensic news analysis conducted on {company}. Due to technical issues during report generation, this is a simplified version of the full analysis.
        
        ## Key Findings
        
        The analysis identified {len(research_results)} significant events related to {company}.
        
        ## Limitation
        
        The full report could not be generated due to technical issues. Please refer to the raw analysis data for complete findings.
        """
        
        state["final_report"] = fallback_report
        print("[Meta Agent Final] Generated fallback report due to errors.")
    
    return {**state, "goto": END}