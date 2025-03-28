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

def generate_final_report(company: str, rag_results: Dict, transcript_results: List[Dict], 
                          corporate_results: Dict, llm) -> str:
    """
    Generate a comprehensive report based on all collected data - passing raw data directly to the LLM
    """
    print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Starting for company: {company}")
    print(f"[Corporate Meta Writer Agent] DEBUG: Input data summary:")
    print(f"[Corporate Meta Writer Agent] DEBUG: - RAG results: {type(rag_results)}, is None: {rag_results is None}")
    if rag_results:
        print(f"[Corporate Meta Writer Agent] DEBUG: - RAG results contains {len(rag_results)} queries")
    print(f"[Corporate Meta Writer Agent] DEBUG: - Transcript results: {type(transcript_results)}, is None: {transcript_results is None}")
    if transcript_results:
        print(f"[Corporate Meta Writer Agent] DEBUG: - Transcript results contains {len(transcript_results)} transcripts")
    print(f"[Corporate Meta Writer Agent] DEBUG: - Corporate results: {type(corporate_results)}, is None: {corporate_results is None}")
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

        with open(os.path.join(debug_dir, "transcript_results.json"), "w") as file:
            json.dump(transcript_results, file, indent=4)
        
        print(f"[Corporate Meta Writer Agent] DEBUG: Saved raw data to JSON files for debugging")
        
        # Generate the report
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Preparing prompt for LLM")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Prepare data for prompt with size limitations
        max_rag_size = 50000
        max_transcript_size = 50000
        max_corporate_size = 70000
        
        # Convert raw data to JSON for prompt, applying size limits if needed
        rag_json = json.dumps(rag_results, indent=2)
        transcript_json = json.dumps(transcript_results, indent=2)
        corporate_json = json.dumps(corporate_results, indent=2)
        
        if len(rag_json) > max_rag_size:
            print(f"[Corporate Meta Writer Agent] WARNING: RAG data too large ({len(rag_json)} chars), truncating to {max_rag_size} chars")
            rag_json = rag_json[:max_rag_size]
        
        if len(transcript_json) > max_transcript_size:
            print(f"[Corporate Meta Writer Agent] WARNING: Transcript data too large ({len(transcript_json)} chars), truncating to {max_transcript_size} chars")
            transcript_json = transcript_json
            
        if len(corporate_json) > max_corporate_size:
            print(f"[Corporate Meta Writer Agent] WARNING: Corporate data too large ({len(corporate_json)} chars), truncating to {max_corporate_size} chars")
            corporate_json = corporate_json
        
        print(f"[Corporate Meta Writer Agent] DEBUG: Data prepared for prompt:")
        print(f"[Corporate Meta Writer Agent] DEBUG: - RAG data: {len(rag_json)} chars")
        print(f"[Corporate Meta Writer Agent] DEBUG: - Transcript data: {len(transcript_json)} chars")
        print(f"[Corporate Meta Writer Agent] DEBUG: - Corporate data: {len(corporate_json)} chars")
        
        # Use prompt templates for generate_final_report
        variables = {
            "company": company,
            "current_date": current_date,
            "rag_json": rag_json,
            "transcript_json": transcript_json,
            "corporate_json": corporate_json
        }
        
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "corporate_meta_writer_agent", 
            "generate_final_report", 
            variables
        )
        
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Constructing messages for LLM")
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Invoking LLM")
        response = llm.invoke(messages)
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - LLM response received")
        
        # Check if response is None
        if response is None:
            print(f"[Corporate Meta Writer Agent] ERROR: LLM response is None")
            raise ValueError("LLM response is None")
            
        # Check if response.content exists and is not None
        if not hasattr(response, 'content') or response.content is None:
            print(f"[Corporate Meta Writer Agent] ERROR: LLM response has no content attribute or content is None")
            raise ValueError("LLM response has no content")
            
        report = response.content.strip()
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Report generated successfully with {len(report)} characters")
        
        # Check if expected sections are in the report
        expected_sections = ["Key Personnel", "Major Announcements", "Conference Calls", "Business Overview"]
        missing_sections = []
        for section in expected_sections:
            if section.lower() not in report.lower() and section not in report:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"[Corporate Meta Writer Agent] WARNING: Report is missing these sections: {missing_sections}")
        
        print(f"[Corporate Meta Writer Agent] Successfully generated report for {company}")
        return report
        
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
        
        # Check if PDF is available for RAG analysis
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
            
            # Go directly to YouTube search
            return {**state, "goto": "corporate_meta_writer_agent"}
    
    # STEP 2: After RAG analysis, route to YouTube agent for searches
    elif current_step == "post_rag":
        print("[Corporate Meta Writer Agent] STEP 2: RAG analysis complete. Preparing YouTube search...")
        
        # Check RAG results
        rag_results = state.get("rag_results", {})
        print(f"[Corporate Meta Writer Agent] DEBUG: RAG results received: {type(rag_results)}, is None: {rag_results is None}")
        if rag_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: RAG results contain {len(rag_results)} queries")
        
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
                    print(f"[Corporate Meta Writer Agent] DEBUG: Transcript {i+1} '{result.get('title', 'Unknown')}' length: {len(transcript) if transcript else 0}")
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
        transcript_results = state.get("transcript_results", [])
        corporate_results = state.get("corporate_results", {})
        
        print(f"[Corporate Meta Writer Agent] DEBUG: Data summary before report generation:")
        print(f"[Corporate Meta Writer Agent] DEBUG: - RAG results: {type(rag_results)}, is None: {rag_results is None}")
        if rag_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: - RAG results contain {len(rag_results)} queries")
        print(f"[Corporate Meta Writer Agent] DEBUG: - Transcript results: {type(transcript_results)}, is None: {transcript_results is None}")
        if transcript_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: - Transcript results contain {len(transcript_results)} transcripts")
        print(f"[Corporate Meta Writer Agent] DEBUG: - Corporate results: {type(corporate_results)}, is None: {corporate_results is None}")
        if corporate_results:
            print(f"[Corporate Meta Writer Agent] DEBUG: - Corporate results contain {len(corporate_results)} entries")
        
        # Generate the final report
        print("[Corporate Meta Writer Agent] STEP 5: Calling generate_final_report function")
        final_report = generate_final_report(
            company, 
            rag_results,
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