import json
import os
import yaml
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
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
    AGENT_CONFIG = LLM_CONFIG.get("corporate_meta_writer_agent", LLM_CONFIG.get("default", {}))
except Exception as e:
    print(f"Error loading LLM config: {e}, using defaults")
    AGENT_CONFIG = {"model": "gemini-2.0-flash", "temperature": 0.2}

from backend.utils.prompt_manager import PromptManager

# Initialize prompt manager with path relative to current file
prompt_manager = PromptManager(os.path.join(BASE_DIR, "prompts"))

def load_preset_queries(preset_file: Optional[str] = None) -> List[str]:
    """
    Load preset queries from a YAML file
    """
    if preset_file is None:
        # Use relative path by default
        preset_file = os.path.join(BASE_DIR, "assets", "preset_queries.yaml")
    try:
        with open(preset_file, "r") as file:
            data = yaml.safe_load(file)
            return data.get("queries", [])
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error loading preset queries: {e}")
        # Return some default queries as fallback
        return [
            "What are the company's key sustainability goals and targets?",
            "How does the company measure and report its ESG performance?",
            "What initiatives has the company taken to reduce its environmental impact?",
            "How does the company manage risks related to climate change and resource scarcity?",
            "What policies does the company have in place for ethical governance and compliance?",
            "How does the company ensure diversity, equity, and inclusion in its workforce?",
            "What are the company's commitments to responsible sourcing and supply chain sustainability?",
            "How does the company engage with stakeholders on ESG issues?",
            "What frameworks or standards does the company follow for ESG reporting?",
            "How does the company integrate ESG considerations into its long-term business strategy?"
        ]


def select_relevant_videos(company: str, search_results: Dict, llm) -> Dict[str, List[Dict]]:
    """
    Use LLM to select the most relevant conference call videos for each quarter - exactly one per quarter
    """
    print(f"[Corporate Meta Writer Agent] Selecting relevant videos for {company}")
    
    quarterly_videos = {
        "Q1": [],
        "Q2": [],
        "Q3": [],
        "Q4": []
    }
    
    try:
        # Process videos with published dates
        all_videos = []
        for query, results in search_results.items():
            for video in results:
                # Clean up the data for the LLM
                all_videos.append({
                    "id": video.get("id", ""),
                    "title": video.get("title", ""),
                    "description": video.get("description", ""),
                    "channel_title": video.get("channel_title", ""),
                    "published_at": video.get("published_at", "")
                })
        
        # Early return if no videos found
        if not all_videos:
            print(f"[Corporate Meta Writer Agent] No videos found for {company}")
            return quarterly_videos
        
        all_videos = json.dumps(all_videos, indent=2)
        # Use prompt templates for select_relevant_videos
        variables = {
            "company": company,
            "all_videos": all_videos
        }
        
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "corporate_meta_writer_agent", 
            "select_relevant_videos", 
            variables
        )
        
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Extract JSON from response if needed
        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_content = content.split("```")[1].strip()
        else:
            json_content = content
        
        selected_videos = json.loads(json_content)
        
        # Log the results
        video_count = sum(len(videos) for videos in selected_videos.values())
        print(f"[Corporate Meta Writer Agent] Selected {video_count} videos (max one per quarter)")
        for quarter, videos in selected_videos.items():
            if videos:
                print(f"[Corporate Meta Writer Agent] - {quarter}: {videos[0].get('title', 'Unknown')}")
            else:
                print(f"[Corporate Meta Writer Agent] - {quarter}: No suitable video found")
        
        return selected_videos
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error selecting videos: {str(e)}")
        return quarterly_videos

def generate_executive_summary(company: str, current_date: str, sections: Dict[str, str], llm) -> str:
    """Generate the executive summary section of the report."""
    print(f"[Corporate Meta Writer Agent] Generating Executive Summary section")
    
    try:
        # Create a summary of available sections for the executive summary
        sections_summary = {}
        for section_name, content in sections.items():
            # Extract first paragraph or two from each section (simplified)
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 2:
                summary = '\n\n'.join(paragraphs[:2])  # Take first two paragraphs
            else:
                summary = content
            sections_summary[section_name] = summary[:1000]  # Limit size
        
        # Convert to JSON for the prompt
        sections_json = json.dumps(sections_summary, indent=2)
        
        # Prepare variables for prompt
        variables = {
            "company": company,
            "current_date": current_date,
            "sections_summary": sections_json
        }
        
        # Get prompts
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "corporate_meta_writer_agent", 
            "executive_summary_section", 
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
        
        print(f"[Corporate Meta Writer Agent] Executive Summary generated: {len(summary)} chars")
        return summary
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error generating Executive Summary: {e}")
        return f"# Executive Summary\n\nThis report provides a comprehensive analysis of {company} based on available data sources including corporate filings, conference call transcripts, and document analysis. The report examines the company's governance structure, business operations, sustainability initiatives, and recent developments.\n\nGenerated on: {current_date}"

def generate_key_personnel_section(company: str, corporate_results: Dict, llm) -> str:
    """Generate the key personnel section of the report."""
    print(f"[Corporate Meta Writer Agent] Generating Key Personnel section")
    
    if not corporate_results or "Key_Personnel" not in corporate_results:
        print(f"[Corporate Meta Writer Agent] No key personnel data found")
        return "# Key Personnel\n\nNo key personnel information available for this company."
    
    try:
        # Extract just the Key_Personnel data
        key_personnel_data = corporate_results.get("Key_Personnel", {})
        
        # Convert to JSON for the prompt
        key_personnel_json = json.dumps(key_personnel_data, indent=2)
        
        # Prepare variables for prompt
        variables = {
            "company": company,
            "key_personnel_data": key_personnel_json
        }
        
        # Get prompts
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "corporate_meta_writer_agent", 
            "key_personnel_section", 
            variables
        )
        
        # Create messages for LLM
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        section_content = response.content.strip()
        
        print(f"[Corporate Meta Writer Agent] Key Personnel section generated: {len(section_content)} chars")
        return section_content
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error generating Key Personnel section: {e}")
        return "# Key Personnel\n\nUnable to generate key personnel information due to technical error."

def generate_business_overview_section(company: str, rag_summaries: Dict, llm) -> str:
    """Generate the business overview section using document analysis results."""
    print(f"[Corporate Meta Writer Agent] Generating Business Overview section")
    
    if not rag_summaries:
        print(f"[Corporate Meta Writer Agent] No RAG summary data found")
        return "# Business Overview\n\nNo document analysis results available for business overview."
    
    try:
        # Find business-related queries
        business_queries = [q for q in rag_summaries.keys() 
                          if any(term in q.lower() for term in 
                                ["business", "operation", "market", "product", "strategy", "model"])]
        
        # If no specific business queries, use all available queries
        if not business_queries:
            business_queries = list(rag_summaries.keys())[:3]  # Use first 3 queries
        
        # Extract summaries for business-related queries
        business_data = {query: rag_summaries[query] for query in business_queries if query in rag_summaries}
        
        # Convert to JSON for the prompt
        business_json = json.dumps(business_data, indent=2)
        
        # Prepare variables for prompt
        variables = {
            "company": company,
            "business_data": business_json
        }
        
        # Get prompts
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "corporate_meta_writer_agent", 
            "business_overview_section", 
            variables
        )
        
        # Create messages for LLM
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        section_content = response.content.strip()
        
        print(f"[Corporate Meta Writer Agent] Business Overview section generated: {len(section_content)} chars")
        return section_content
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error generating Business Overview section: {e}")
        return "# Business Overview\n\nUnable to generate business overview information due to technical error."

def generate_esg_section(company: str, rag_summaries: Dict, llm) -> str:
    """Generate the ESG/sustainability section using document analysis results."""
    print(f"[Corporate Meta Writer Agent] Generating ESG/Sustainability section")
    
    if not rag_summaries:
        print(f"[Corporate Meta Writer Agent] No RAG summary data found for ESG")
        return "# Sustainability & ESG\n\nNo document analysis results available for sustainability and ESG information."
    
    try:
        # Find ESG-related queries
        esg_queries = [q for q in rag_summaries.keys() 
                     if any(term in q.lower() for term in 
                           ["sustainability", "esg", "environmental", "social", "governance", 
                            "climate", "carbon", "emission", "diversity", "ethical"])]
        
        # If no specific ESG queries, look for relevant terms in other query results
        if not esg_queries:
            for query, summary in rag_summaries.items():
                if any(term in summary.lower() for term in 
                      ["sustainability", "esg", "environmental", "social", "governance", 
                       "climate", "carbon", "emission", "diversity", "ethical"]):
                    esg_queries.append(query)
        
        # Still no ESG queries, use generic ones if available
        if not esg_queries and rag_summaries:
            esg_queries = list(rag_summaries.keys())[:2]  # Use first 2 queries as fallback
        
        # Extract summaries for ESG-related queries
        esg_data = {query: rag_summaries[query] for query in esg_queries if query in rag_summaries}
        
        # Convert to JSON for the prompt
        esg_json = json.dumps(esg_data, indent=2)
        
        # Prepare variables for prompt
        variables = {
            "company": company,
            "esg_data": esg_json
        }
        
        # Get prompts
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "corporate_meta_writer_agent", 
            "esg_section", 
            variables
        )
        
        # Create messages for LLM
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        section_content = response.content.strip()
        
        print(f"[Corporate Meta Writer Agent] ESG section generated: {len(section_content)} chars")
        return section_content
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error generating ESG section: {e}")
        return "# Sustainability & ESG\n\nUnable to generate sustainability and ESG information due to technical error."

def generate_announcements_section(company: str, corporate_results: Dict, llm) -> str:
    """Generate the major announcements section using corporate data."""
    print(f"[Corporate Meta Writer Agent] Generating Major Announcements section")
    
    announcements_data = {}
    
    # Extract relevant data from corporate results
    if corporate_results:
        # Include all corporate data EXCEPT Key_Personnel
        for key, value in corporate_results.items():
            if key != "Key_Personnel":
                announcements_data[key] = value
    
    if not announcements_data:
        print(f"[Corporate Meta Writer Agent] No announcements data found")
        return "# Major Announcements\n\nNo corporate announcements data available."
    
    try:
        # Convert to JSON for the prompt
        announcements_json = json.dumps(announcements_data, indent=2)
        
        # Prepare variables for prompt
        variables = {
            "company": company,
            "announcements_data": announcements_json
        }
        
        # Get prompts
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "corporate_meta_writer_agent", 
            "announcements_section", 
            variables
        )
        
        # Create messages for LLM
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        section_content = response.content.strip()
        
        print(f"[Corporate Meta Writer Agent] Announcements section generated: {len(section_content)} chars")
        return section_content
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error generating Announcements section: {e}")
        return "# Major Announcements\n\nUnable to generate corporate announcements information due to technical error."

def generate_conference_calls_section(company: str, transcript_results: List[Dict], llm) -> str:
    """Generate the conference calls section using transcript data."""
    print(f"[Corporate Meta Writer Agent] Generating Conference Calls section")
    
    if not transcript_results:
        print(f"[Corporate Meta Writer Agent] No transcript data found")
        return "# Conference Calls\n\nNo conference call transcript data available."
    
    try:
        # Extract just the needed information - use summaries instead of full transcripts
        summarized_data = []
        for result in transcript_results:
            summarized_data.append({
                "title": result.get("title", "Untitled Call"),
                "id": result.get("id", ""),
                # Use transcript_summary instead of full transcript
                "summary": result.get("transcript_summary", "No summary available.")
            })
        
        # Convert to JSON for the prompt
        transcript_json = json.dumps(summarized_data, indent=2)
        
        # Prepare variables for prompt
        variables = {
            "company": company,
            "transcript_data": transcript_json
        }
        
        # Get prompts
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "corporate_meta_writer_agent", 
            "conference_calls_section", 
            variables
        )
        
        # Create messages for LLM
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        section_content = response.content.strip()
        
        print(f"[Corporate Meta Writer Agent] Conference Calls section generated: {len(section_content)} chars")
        return section_content
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error generating Conference Calls section: {e}")
        return "# Conference Calls\n\nUnable to generate conference call information due to technical error."

def generate_governance_concerns_section(company: str, corporate_results: Dict, transcript_results: List[Dict], rag_summaries: Dict, llm) -> str:
    """Generate the governance concerns section using data from all sources."""
    print(f"[Corporate Meta Writer Agent] Generating Governance Concerns section")
    
    if not (corporate_results or transcript_results or rag_summaries):
        print(f"[Corporate Meta Writer Agent] No data found for governance concerns")
        return "# Major Governance Concerns\n\nNo data available to analyze governance concerns."
    
    try:
        # Create a combined dataset with key information from all sources
        combined_data = {
            "corporate_governance": {},
            "transcript_concerns": [],
            "document_concerns": {}
        }
        
        # Extract key governance data from corporate results
        if corporate_results and "Key_Personnel" in corporate_results:
            combined_data["corporate_governance"] = {
                "board_structure": corporate_results["Key_Personnel"].get("board_of_directors", []),
                "committees": corporate_results["Key_Personnel"].get("communities", {})
            }
        
        # Extract governance mentions from transcript summaries
        if transcript_results:
            for result in transcript_results:
                summary = result.get("transcript_summary", "")
                if any(term in summary.lower() for term in ["governance", "board", "compliance", "ethics",
                                                          "regulatory", "risk", "audit", "oversight"]):
                    combined_data["transcript_concerns"].append({
                        "title": result.get("title", "Untitled Call"),
                        "governance_content": summary
                    })
        
        # Extract governance data from RAG summaries
        if rag_summaries:
            governance_queries = {}
            for query, summary in rag_summaries.items():
                if any(term in query.lower() or term in summary.lower() 
                      for term in ["governance", "board", "compliance", "ethics", "regulatory", 
                                  "risk", "audit", "oversight", "committee"]):
                    governance_queries[query] = summary
            combined_data["document_concerns"] = governance_queries
        
        # Convert to JSON for the prompt
        combined_json = json.dumps(combined_data, indent=2)
        
        # Prepare variables for prompt
        variables = {
            "company": company,
            "governance_data": combined_json
        }
        
        # Get prompts
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "corporate_meta_writer_agent", 
            "governance_concerns_section", 
            variables
        )
        
        # Create messages for LLM
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        section_content = response.content.strip()
        
        print(f"[Corporate Meta Writer Agent] Governance Concerns section generated: {len(section_content)} chars")
        return section_content
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error generating Governance Concerns section: {e}")
        return "# Major Governance Concerns\n\nUnable to generate governance concerns information due to technical error."

def generate_final_report(company: str, rag_results: Dict, rag_summaries: Dict, transcript_results: List[Dict], 
                          corporate_results: Dict, llm) -> str:
    """
    Generate a comprehensive report based on all collected data using a section-by-section approach.
    """
    print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Starting for company: {company}")
    print(f"[Corporate Meta Writer Agent] DEBUG: Input data summary:")
    print(f"[Corporate Meta Writer Agent] DEBUG: - RAG results: {type(rag_results)}, is None: {rag_results is None}")
    print(f"[Corporate Meta Writer Agent] DEBUG: - RAG summaries: {type(rag_summaries)}, is None: {rag_summaries is None}")
    print(f"[Corporate Meta Writer Agent] DEBUG: - Transcript results: {type(transcript_results)}, is None: {transcript_results is None}")
    print(f"[Corporate Meta Writer Agent] DEBUG: - Corporate results: {type(corporate_results)}, is None: {corporate_results is None}")
    
    if rag_results:
        print(f"[Corporate Meta Writer Agent] DEBUG: - RAG results contains {len(rag_results)} queries")
    if rag_summaries:
        print(f"[Corporate Meta Writer Agent] DEBUG: - RAG summaries contains {len(rag_summaries)} queries")
    if transcript_results:
        print(f"[Corporate Meta Writer Agent] DEBUG: - Transcript results contains {len(transcript_results)} transcripts")
    if corporate_results:
        print(f"[Corporate Meta Writer Agent] DEBUG: - Corporate results contains {len(corporate_results)} entries")
    
    try:
        # Save raw data to JSON files for debugging
        debug_dir = os.path.join(BASE_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        with open(os.path.join(debug_dir, "corporate_data_temp.json"), "w") as file:
            json.dump(corporate_results, file, indent=4)
        
        with open(os.path.join(debug_dir, "rag_results.json"), "w") as file:
            json.dump(rag_results, file, indent=4)
            
        with open(os.path.join(debug_dir, "rag_summaries.json"), "w") as file:
            json.dump(rag_summaries, file, indent=4)

        with open(os.path.join(debug_dir, "transcript_results.json"), "w") as file:
            json.dump(transcript_results, file, indent=4)
        
        print(f"[Corporate Meta Writer Agent] DEBUG: Saved raw data to JSON files for debugging")
        
        # Current date for report
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # STEP 1: Generate individual sections
        report_sections = {}
        
        # Generate Key Personnel section
        print(f"[Corporate Meta Writer Agent] STEP: Generating Key Personnel section")
        key_personnel_section = generate_key_personnel_section(company, corporate_results, llm)
        report_sections["key_personnel"] = key_personnel_section
        
        # Generate Business Overview section
        print(f"[Corporate Meta Writer Agent] STEP: Generating Business Overview section")
        business_overview_section = generate_business_overview_section(company, rag_summaries, llm)
        report_sections["business_overview"] = business_overview_section
        
        # Generate ESG section
        print(f"[Corporate Meta Writer Agent] STEP: Generating ESG section")
        esg_section = generate_esg_section(company, rag_summaries, llm)
        report_sections["esg"] = esg_section
        
        # Generate Announcements section
        print(f"[Corporate Meta Writer Agent] STEP: Generating Announcements section")
        announcements_section = generate_announcements_section(company, corporate_results, llm)
        report_sections["announcements"] = announcements_section
        
        # Generate Conference Calls section
        print(f"[Corporate Meta Writer Agent] STEP: Generating Conference Calls section")
        conference_calls_section = generate_conference_calls_section(company, transcript_results, llm)
        report_sections["conference_calls"] = conference_calls_section
        
        # Generate Governance Concerns section
        print(f"[Corporate Meta Writer Agent] STEP: Generating Governance Concerns section")
        governance_concerns_section = generate_governance_concerns_section(
            company, corporate_results, transcript_results, rag_summaries, llm
        )
        report_sections["governance_concerns"] = governance_concerns_section
        
        # Generate Executive Summary (last, after other sections are done)
        print(f"[Corporate Meta Writer Agent] STEP: Generating Executive Summary")
        executive_summary = generate_executive_summary(company, current_date, report_sections, llm)
        report_sections["executive_summary"] = executive_summary
        
        # STEP 2: Combine all sections into final report
        print(f"[Corporate Meta Writer Agent] STEP: Assembling final report")
        
        # Define section order
        section_order = [
            "executive_summary", 
            "key_personnel", 
            "business_overview", 
            "esg", 
            "announcements", 
            "conference_calls", 
            "governance_concerns"
        ]
        
        # Combine sections in order
        report_parts = [f"# {company} Analysis Report\n\nReport Date: {current_date}\n"]
        
        for section_name in section_order:
            if section_name in report_sections and report_sections[section_name]:
                # Get section content
                section_content = report_sections[section_name]
                
                # Check if section already has a header
                if not section_content.startswith('#'):
                    # Add header with proper level
                    section_title = section_name.replace('_', ' ').title()
                    section_content = f"## {section_title}\n\n{section_content}"
                
                # Add to report
                report_parts.append(section_content)
        
        # Join all parts with separators
        final_report = "\n\n".join(report_parts)
        
        print(f"[Corporate Meta Writer Agent] STEP: Final report assembled with {len(final_report)} characters")
        
        # Check if expected sections are in the report
        expected_sections = ["Key Personnel", "Major Announcements", "Conference Calls", "Business Overview"]
        missing_sections = []
        for section in expected_sections:
            if section.lower() not in final_report.lower() and section not in final_report:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"[Corporate Meta Writer Agent] WARNING: Report is missing these sections: {missing_sections}")
        
        print(f"[Corporate Meta Writer Agent] Successfully generated report for {company}")
        return final_report
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] ERROR: Error generating report: {str(e)}")
        import traceback
        print(f"[Corporate Meta Writer Agent] ERROR: Traceback: {traceback.format_exc()}")
        
        return f"""
        # {company} Analysis Report
        
        ## Error Generating Complete Report
        
        Unfortunately, an error occurred while generating the complete analysis report. This may be due to:
        
        - Complexity of the data
        - Insufficient information available
        - Technical processing issue
        
        ### Available Information
        
        - Document Analysis: {"Available" if rag_results and not isinstance(rag_results, type(None)) else "Not available"}
        - Conference Call Transcripts: {len(transcript_results) if transcript_results and not isinstance(transcript_results, type(None)) else 0} transcripts found
        - Corporate Data: {"Available" if corporate_results and not isinstance(corporate_results, type(None)) else "Not available"}
        
        ### Error Details
        
        ```
        {str(e)}
        ```
        
        Please try again with additional information or contact technical support.
        
        Generated on: {datetime.now().strftime("%Y-%m-%d")}
        """

def corporate_meta_writer_agent(state: Dict) -> Dict:
    """
    Orchestrates the workflow between rag_agent, youtube_agent, and corporate_agent
    to generate a comprehensive corporate analysis report.
    """
    print("[Corporate Meta Writer Agent] MAIN: Starting workflow...")
    
    # Extract key information from state
    company = state.get("company", "")
    company_symbol = state.get("company_symbol", "")
    pdf_path = state.get("pdf_path", "")
    
    print(f"[Corporate Meta Writer Agent] DEBUG: Company: '{company}', Symbol: '{company_symbol}', PDF: '{pdf_path}'")
    
    # Validate required inputs
    if not company:
        print("[Corporate Meta Writer Agent] ERROR: Company name is missing!")
        return {**state, "goto": "END", "error": "Company name is required"}
    
    # PDF is now optional - just log a warning if not found
    if pdf_path and not os.path.exists(pdf_path):
        print(f"[Corporate Meta Writer Agent] WARNING: PDF file not found at {pdf_path}, continuing without document analysis")
        pdf_path = None
        state["pdf_path"] = None
    
    # Initialize LLM
    print("[Corporate Meta Writer Agent] MAIN: Initializing LLM")
    try:
        llm = ChatGoogleGenerativeAI(
            model=AGENT_CONFIG["model"], 
            temperature=AGENT_CONFIG["temperature"]
        )
        print("[Corporate Meta Writer Agent] DEBUG: LLM initialized successfully")
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] ERROR: Failed to initialize LLM: {e}")
        return {**state, "goto": "END", "error": f"Failed to initialize LLM: {str(e)}"}
    
    # Track workflow state
    current_step = state.get("corporate_meta_step", "start")
    print(f"[Corporate Meta Writer Agent] MAIN: Current step: '{current_step}'")
    
    # STEP 1: Initial setup and route to RAG agent if PDF is available
    if current_step == "start":
        print("[Corporate Meta Writer Agent] STEP 1: Loading preset queries and preparing RAG analysis...")
        
        # Load preset queries
        try:
            queries = load_preset_queries()
            print(f"[Corporate Meta Writer Agent] DEBUG: Loaded {len(queries)} preset queries")
        except Exception as e:
            print(f"[Corporate Meta Writer Agent] ERROR: Failed to load preset queries: {str(e)}")
            queries = [
                "What are the key financial highlights?",
                "Describe the company's business model",
                "What risks does the company face?",
                "What is the company's growth strategy?"
            ]
            print(f"[Corporate Meta Writer Agent] DEBUG: Using fallback queries")
        
        # Update state for next step
        state["corporate_meta_step"] = "post_rag"
        
        # Check if PDF path exists for RAG analysis
        if pdf_path:
            # Update state for RAG agent
            state["rag_queries"] = queries
            state["rag_pdf_path"] = pdf_path
            state["is_file_embedded"] = False
            
            print(f"[Corporate Meta Writer Agent] STEP 1: Routing to rag_agent with {len(queries)} queries")
            return {**state, "goto": "rag_agent"}
        else:
            # Skip RAG analysis if no PDF
            print("[Corporate Meta Writer Agent] STEP 1: No PDF provided, skipping RAG analysis")
            state["rag_results"] = {}
            state["rag_summaries"] = {}
            
            # Go directly to YouTube search
            return {**state, "goto": "corporate_meta_writer_agent"}
    
    # STEP 2: After RAG analysis, route to YouTube agent for searches
    elif current_step == "post_rag":
        print("[Corporate Meta Writer Agent] STEP 2: RAG analysis complete. Preparing YouTube search...")
        
        # Check RAG results
        rag_results = state.get("rag_results", {})
        rag_summaries = state.get("rag_summaries", {})  # Get the new summaries
        
        print(f"[Corporate Meta Writer Agent] DEBUG: RAG results received: {type(rag_results)}, is None: {rag_results is None}")
        print(f"[Corporate Meta Writer Agent] DEBUG: RAG summaries received: {type(rag_summaries)}, is None: {rag_summaries is None}")
        
        if rag_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: RAG results contain {len(rag_results)} queries")
        if rag_summaries:
            print(f"[Corporate Meta Writer Agent] DEBUG: RAG summaries contain {len(rag_summaries)} summaries")
        
        # Create conference call search queries
        search_queries = [
            f"conference call {company} FY 25 Q3",
            f"conference call {company} FY 25 Q2",
            f"conference call {company} FY 25 Q1",
            f"conference call {company} FY 24 Q4",
            f"earnings call {company} FY 25 Q3",
            f"earnings call {company} FY 25 Q2",
            f"earnings call {company} FY 25 Q1",
            f"earnings call {company} FY 24 Q4"
        ]
        
        # Update state for YouTube agent
        state["youtube_agent_action"] = "search"
        state["search_queries"] = search_queries
        state["corporate_meta_step"] = "select_videos"
        
        print(f"[Corporate Meta Writer Agent] STEP 2: Routing to youtube_agent for search with {len(search_queries)} queries")
        return {**state, "goto": "youtube_agent"}
    
    # STEP 3: Process YouTube search results and select relevant videos
    elif current_step == "select_videos":
        print("[Corporate Meta Writer Agent] STEP 3: YouTube search complete. Selecting relevant videos...")
        
        search_results = state.get("search_results", {})
        print(f"[Corporate Meta Writer Agent] DEBUG: Search results received: {type(search_results)}, is None: {search_results is None}")
        if search_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: Search results contain {len(search_results)} queries")
            for query, results in search_results.items():
                if results is None:
                    print(f"[Corporate Meta Writer Agent] WARNING: Results for query '{query}' is None")
                else:
                    print(f"[Corporate Meta Writer Agent] DEBUG: Query '{query}' has {len(results)} results")
        
        if not search_results:
            print("[Corporate Meta Writer Agent] WARNING: No YouTube search results found")
            state["selected_videos"] = []
            state["corporate_meta_step"] = "get_corporate_data"
            print("[Corporate Meta Writer Agent] STEP 3: No videos to process, routing to corporate_agent")
            return {**state, "goto": "corporate_agent"}
        
        # Use LLM to select relevant videos for each quarter
        print("[Corporate Meta Writer Agent] STEP 3: Selecting relevant videos with LLM")
        try:
            selected_videos = select_relevant_videos(company, search_results, llm)
            print(f"[Corporate Meta Writer Agent] DEBUG: Selected videos: {type(selected_videos)}, is None: {selected_videos is None}")
            if selected_videos:
                for quarter, videos in selected_videos.items():
                    print(f"[Corporate Meta Writer Agent] DEBUG: Quarter '{quarter}' has {len(videos)} videos")
        except Exception as e:
            print(f"[Corporate Meta Writer Agent] ERROR: Failed to select relevant videos: {str(e)}")
            import traceback
            print(f"[Corporate Meta Writer Agent] ERROR: Traceback: {traceback.format_exc()}")
            selected_videos = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        
        # Prepare video list for transcription
        video_ids = []
        for quarter, videos in selected_videos.items():
            for video in videos:
                if "id" in video and "title" in video:
                    video_ids.append({"id": video["id"], "title": video["title"]})
                else:
                    print(f"[Corporate Meta Writer Agent] WARNING: Video missing required fields: id present: {'id' in video}, title present: {'title' in video}")
        
        state["selected_videos"] = selected_videos
        
        # If videos found, route to YouTube for transcription
        if video_ids:
            state["youtube_agent_action"] = "transcribe"
            state["video_ids"] = video_ids
            state["corporate_meta_step"] = "post_transcription"
            
            print(f"[Corporate Meta Writer Agent] STEP 3: Routing to youtube_agent for transcription of {len(video_ids)} videos")
            return {**state, "goto": "youtube_agent"}
        else:
            # Skip transcription if no videos found
            print("[Corporate Meta Writer Agent] STEP 3: No relevant videos found. Skipping transcription.")
            state["transcript_results"] = []
            state["corporate_meta_step"] = "get_corporate_data"
            
            print("[Corporate Meta Writer Agent] STEP 3: Routing to corporate_agent")
            return {**state, "goto": "corporate_agent"}
    
    # STEP 4: After transcription, route to corporate agent
    elif current_step == "post_transcription":
        print("[Corporate Meta Writer Agent] STEP 4: Transcription complete. Getting corporate data...")
        
        # Check transcript results
        transcript_results = state.get("transcript_results", [])
        print(f"[Corporate Meta Writer Agent] DEBUG: Transcript results received: {type(transcript_results)}, is None: {transcript_results is None}")
        if transcript_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: Transcript results contain {len(transcript_results)} transcripts")
            for i, result in enumerate(transcript_results):
                if "title" in result and "transcript" in result:
                    transcript = result.get("transcript", "")
                    summary = result.get("transcript_summary", "")
                    print(f"[Corporate Meta Writer Agent] DEBUG: Transcript {i+1} '{result.get('title', 'Unknown')}' - full transcript length: {len(transcript) if transcript else 0}, summary length: {len(summary) if summary else 0}")
                else:
                    print(f"[Corporate Meta Writer Agent] WARNING: Transcript {i+1} missing required fields: title present: {'title' in result}, transcript present: {'transcript' in result}")
        
        state["corporate_meta_step"] = "generate_report"
        
        print("[Corporate Meta Writer Agent] STEP 4: Routing to corporate_agent")
        return {**state, "goto": "corporate_agent"}
    
    # STEP 5: After all data collection, generate final report
    elif current_step == "generate_report":
        print("[Corporate Meta Writer Agent] STEP 5: All data collection complete. Starting report generation...")
        
        # Extract all collected data
        rag_results = state.get("rag_results", {})
        rag_summaries = state.get("rag_summaries", {})  # Get the summaries
        transcript_results = state.get("transcript_results", [])
        corporate_results = state.get("corporate_results", {})
        
        print(f"[Corporate Meta Writer Agent] DEBUG: Data summary before report generation:")
        print(f"[Corporate Meta Writer Agent] DEBUG: - RAG results: {type(rag_results)}, is None: {rag_results is None}")
        print(f"[Corporate Meta Writer Agent] DEBUG: - RAG summaries: {type(rag_summaries)}, is None: {rag_summaries is None}")
        if rag_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: - RAG results contain {len(rag_results)} queries")
        if rag_summaries:
            print(f"[Corporate Meta Writer Agent] DEBUG: - RAG summaries contain {len(rag_summaries)} summaries")
        print(f"[Corporate Meta Writer Agent] DEBUG: - Transcript results: {type(transcript_results)}, is None: {transcript_results is None}")
        if transcript_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: - Transcript results contain {len(transcript_results)} transcripts")
        print(f"[Corporate Meta Writer Agent] DEBUG: - Corporate results: {type(corporate_results)}, is None: {corporate_results is None}")
        if corporate_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: - Corporate results contain {len(corporate_results)} entries")
        
        # Generate the final report using our new section-by-section approach
        print("[Corporate Meta Writer Agent] STEP 5: Calling generate_final_report function")
        final_report = generate_final_report(
            company, 
            rag_results,
            rag_summaries,
            transcript_results,
            corporate_results,
            llm
        )
        
        print(f"[Corporate Meta Writer Agent] DEBUG: Final report received, length: {len(final_report) if final_report else 0}")
        
        # Update state with final report
        state["final_report"] = final_report
        state["corporate_meta_status"] = "DONE"
        
        print("[Corporate Meta Writer Agent] STEP 5: Workflow complete. Report generated.")
        return {**state, "goto": "END"}
    
    else:
        print(f"[Corporate Meta Writer Agent] ERROR: Unknown step '{current_step}'")
        return {**state, "goto": "END", "error": f"Unknown workflow step: {current_step}"}