import json
import os
import yaml
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

def load_preset_queries(preset_file: str = "assets/preset_queries.yaml") -> List[str]:
    """
    Load preset queries from a YAML file
    """
    try:
        with open(preset_file, "r") as file:
            data = yaml.safe_load(file)
            return data.get("queries", [])
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error loading preset queries: {e}")
        # Return some default queries as fallback
        return [
            "What are the key financial highlights?",
            "Describe the company's business model",
            "What risks does the company face?",
            "What is the company's growth strategy?"
        ]

def select_relevant_videos(company: str, search_results: Dict, llm) -> Dict[str, List[Dict]]:
    """
    Use LLM to select the most relevant conference call videos for each quarter
    """
    print(f"[Corporate Meta Writer Agent] Selecting relevant videos for {company}")
    
    quarterly_videos = {
        "Q1": [],
        "Q2": [],
        "Q3": [],
        "Q4": []
    }
    
    try:
        # Filter search results by quarter
        quarter_keywords = {
            "Q1": ["Q1", "first quarter"],
            "Q2": ["Q2", "second quarter"],
            "Q3": ["Q3", "third quarter"],
            "Q4": ["Q4", "fourth quarter", "Q4", "annual results"]
        }
        
        # Prepare the prompt for the LLM
        all_videos = []
        for query, results in search_results.items():
            for video in results:
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
        
        # Use LLM to select relevant videos
        prompt = f"""
        You are tasked with identifying the most relevant conference or earnings call videos for {company} by quarter.
        
        Here are YouTube search results for conference calls and earnings calls:
        {json.dumps(all_videos, indent=2)}
        
        For each quarter (Q1, Q2, Q3, Q4), identify up to 4 most relevant videos about {company}'s earnings or conference calls.
        Use these criteria:
        1. Title explicitly mentions the quarter
        2. Published by official company channel or financial news sources
        3. More recent videos preferred
        4. Most directly relevant to financial results/earnings calls
        
        Return a JSON object with this structure:
        {{
          "Q1": [{{id: "video_id1", title: "title1"}}, ...],
          "Q2": [{{id: "video_id2", title: "title2"}}, ...],
          "Q3": [{{id: "video_id3", title: "title3"}}, ...],
          "Q4": [{{id: "video_id4", title: "title4"}}, ...]
        }}
        
        Include up to 4 videos per quarter, fewer if not enough relevant videos found.
        """
        
        messages = [
            ("system", "You are an AI assistant that helps identify relevant financial videos for companies."),
            ("human", prompt)
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
        print(f"[Corporate Meta Writer Agent] Selected {sum(len(v) for v in selected_videos.values())} videos across all quarters")
        
        return selected_videos
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error selecting videos: {str(e)}")
        return quarterly_videos

def generate_final_report(company: str, rag_results: Dict, transcript_results: List[Dict], 
                          corporate_results: Dict, llm) -> str:
    """
    Generate a comprehensive report based on all collected data
    """
    print(f"[Corporate Meta Writer Agent] Generating final report for {company}")
    
    try:
        # Prepare transcript summaries
        transcript_summaries = []
        for result in transcript_results:
            if "transcript" in result and "title" in result:
                # Truncate long transcripts for the prompt
                transcript = result["transcript"]
                if len(transcript) > 2000:
                    transcript = transcript[:2000] + "... [truncated]"
                
                transcript_summaries.append({
                    "title": result["title"],
                    "transcript_summary": transcript
                })
        
        # Prepare RAG results
        rag_findings = []
        for query, results in rag_results.items():
            relevant_texts = []
            for i, result in enumerate(results[:3]):  # Take top 3 results per query
                relevant_texts.append(result.get("text", "")[:500])  # Truncate long texts
            
            rag_findings.append({
                "query": query,
                "findings": relevant_texts
            })
        
        # Summarize corporate results
        corporate_summary = {}
        for data_type, data in corporate_results.items():
            if isinstance(data, list) and data:
                corporate_summary[data_type] = f"{len(data)} entries found"
            elif isinstance(data, dict) and data:
                corporate_summary[data_type] = f"{len(data)} entries found"
            else:
                corporate_summary[data_type] = "No data available"
        
        # Generate the report
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
        You are a financial analyst creating a comprehensive report about {company} based on multiple data sources.
        
        Today's date: {current_date}
        
        Document Analysis Results:
        {json.dumps(rag_findings, indent=2)}
        
        Conference Call Information:
        {json.dumps(transcript_summaries, indent=2)}
        
        NSE Corporate Data:
        {json.dumps(corporate_summary, indent=2)}
        
        Generate a comprehensive corporate analysis report with these sections:
        
        1. Executive Summary
        2. Business Overview
        3. Financial Performance Analysis
        4. Key Statements from Management
        5. Risk Factors
        6. Corporate Actions and Events
        7. Future Outlook
        8. Recommendation
        
        Format the report in Markdown. Make it detailed, insightful, and professional.
        """
        
        messages = [
            ("system", "You are an expert financial analyst specializing in comprehensive corporate research."),
            ("human", prompt)
        ]
        
        response = llm.invoke(messages)
        report = response.content.strip()
        
        print(f"[Corporate Meta Writer Agent] Successfully generated report for {company}")
        return report
        
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] Error generating report: {str(e)}")
        return f"""
        # {company} Analysis Report
        
        ## Error Generating Complete Report
        
        Unfortunately, an error occurred while generating the complete analysis report. This may be due to:
        
        - Complexity of the data
        - Insufficient information available
        - Technical processing issue
        
        ### Available Information
        
        - Document Analysis: {"Available" if rag_results else "Not available"}
        - Conference Call Transcripts: {len(transcript_results) if transcript_results else 0} transcripts found
        - Corporate Data: {"Available" if corporate_results else "Not available"}
        
        Please try again with additional information or contact technical support.
        
        Generated on: {datetime.now().strftime("%Y-%m-%d")}
        """

def corporate_meta_writer_agent(state: Dict) -> Dict:
    """
    Orchestrates the workflow between rag_agent, youtube_agent, and corporate_agent
    to generate a comprehensive corporate analysis report.
    """
    print("[Corporate Meta Writer Agent] Starting workflow...")
    
    # Extract key information from state
    company = state.get("company", "")
    company_symbol = state.get("company_symbol", "")
    pdf_path = state.get("pdf_path", "")
    
    # Validate required inputs
    if not company:
        print("[Corporate Meta Writer Agent] ERROR: Company name is missing!")
        return {**state, "goto": "END", "error": "Company name is required"}
    
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"[Corporate Meta Writer Agent] ERROR: PDF file not found at {pdf_path}")
        return {**state, "goto": "END", "error": f"PDF file not found at {pdf_path}"}
    
    # Initialize LLM
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] ERROR: Failed to initialize LLM: {e}")
        return {**state, "goto": "END", "error": f"Failed to initialize LLM: {str(e)}"}
    
    # Track workflow state
    current_step = state.get("corporate_meta_step", "start")
    print(f"[Corporate Meta Writer Agent] Current step: {current_step}")
    
    # STEP 1: Initial setup and route to RAG agent
    if current_step == "start":
        print("[Corporate Meta Writer Agent] Loading preset queries and preparing RAG analysis...")
        
        # Load preset queries
        queries = load_preset_queries()
        
        # Update state for RAG agent
        state["rag_queries"] = queries
        state["rag_pdf_path"] = pdf_path
        state["is_file_embedded"] = False
        state["corporate_meta_step"] = "post_rag"
        
        print(f"[Corporate Meta Writer Agent] Routing to rag_agent with {len(queries)} queries")
        return {**state, "goto": "rag_agent"}
    
    # STEP 2: After RAG analysis, route to YouTube agent for searches
    elif current_step == "post_rag":
        print("[Corporate Meta Writer Agent] RAG analysis complete. Preparing YouTube search...")
        
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
        
        print(f"[Corporate Meta Writer Agent] Routing to youtube_agent for search with {len(search_queries)} queries")
        return {**state, "goto": "youtube_agent"}
    
    # STEP 3: Process YouTube search results and select relevant videos
    elif current_step == "select_videos":
        print("[Corporate Meta Writer Agent] YouTube search complete. Selecting relevant videos...")
        
        search_results = state.get("search_results", {})
        
        if not search_results:
            print("[Corporate Meta Writer Agent] WARNING: No YouTube search results found")
            state["selected_videos"] = []
            state["corporate_meta_step"] = "get_corporate_data"
            return {**state, "goto": "corporate_agent"}
        
        # Use LLM to select relevant videos for each quarter
        selected_videos = select_relevant_videos(company, search_results, llm)
        
        # Prepare video list for transcription
        video_ids = []
        for quarter, videos in selected_videos.items():
            for video in videos:
                if "id" in video and "title" in video:
                    video_ids.append({"id": video["id"], "title": video["title"]})
        
        state["selected_videos"] = selected_videos
        
        # If videos found, route to YouTube for transcription
        if video_ids:
            state["youtube_agent_action"] = "transcribe"
            state["video_ids"] = video_ids
            state["corporate_meta_step"] = "post_transcription"
            
            print(f"[Corporate Meta Writer Agent] Routing to youtube_agent for transcription of {len(video_ids)} videos")
            return {**state, "goto": "youtube_agent"}
        else:
            # Skip transcription if no videos found
            print("[Corporate Meta Writer Agent] No relevant videos found. Skipping transcription.")
            state["transcript_results"] = []
            state["corporate_meta_step"] = "get_corporate_data"
            
            print("[Corporate Meta Writer Agent] Routing to corporate_agent")
            return {**state, "goto": "corporate_agent"}
    
    # STEP 4: After transcription, route to corporate agent
    elif current_step == "post_transcription":
        print("[Corporate Meta Writer Agent] Transcription complete. Getting corporate data...")
        
        state["corporate_meta_step"] = "generate_report"
        
        print("[Corporate Meta Writer Agent] Routing to corporate_agent")
        return {**state, "goto": "corporate_agent"}
    
    # STEP 5: After all data collection, generate final report
    elif current_step == "generate_report":
        print("[Corporate Meta Writer Agent] All data collection complete. Generating final report...")
        
        # Extract all collected data
        rag_results = state.get("rag_results", {})
        transcript_results = state.get("transcript_results", [])
        corporate_results = state.get("corporate_results", {})
        
        # Generate the final report
        final_report = generate_final_report(
            company, 
            rag_results,
            transcript_results,
            corporate_results,
            llm
        )
        
        # Update state with final report
        state["final_report"] = final_report
        state["corporate_meta_status"] = "DONE"
        
        print("[Corporate Meta Writer Agent] Workflow complete. Report generated.")
        return {**state, "goto": "END"}
    
    else:
        print(f"[Corporate Meta Writer Agent] ERROR: Unknown step '{current_step}'")
        return {**state, "goto": "END", "error": f"Unknown workflow step: {current_step}"}