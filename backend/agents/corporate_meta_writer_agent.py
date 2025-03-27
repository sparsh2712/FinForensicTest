import json
import os
import yaml
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

def load_preset_queries(preset_file: str = "/Users/sparsh/Desktop/FinForensicTest/backend/assets/preset_queries.yaml") -> List[str]:
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
            "What are the company’s key sustainability goals and targets?",
            "How does the company measure and report its ESG performance?",
            "What initiatives has the company taken to reduce its environmental impact?",
            "How does the company manage risks related to climate change and resource scarcity?",
            "What policies does the company have in place for ethical governance and compliance?",
            "How does the company ensure diversity, equity, and inclusion in its workforce?",
            "What are the company’s commitments to responsible sourcing and supply chain sustainability?",
            "How does the company engage with stakeholders on ESG issues?",
            "What frameworks or standards does the company follow for ESG reporting?",
            "How does the company integrate ESG considerations into its long-term business strategy?"
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
        # Prepare transcript summaries
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Preparing transcript summaries")
        transcript_summaries = []
        
        # Validate transcript_results
        if transcript_results is None:
            print(f"[Corporate Meta Writer Agent] WARNING: transcript_results is None, initializing as empty list")
            transcript_results = []
            
        for i, result in enumerate(transcript_results):
            print(f"[Corporate Meta Writer Agent] DEBUG: Processing transcript {i+1}/{len(transcript_results)}")
            if "transcript" in result and "title" in result:
                transcript = result["transcript"]
                
                # Check if transcript is None
                if transcript is None:
                    print(f"[Corporate Meta Writer Agent] WARNING: Transcript {i+1} is None for title: {result.get('title', 'Unknown')}")
                    continue
                    
                print(f"[Corporate Meta Writer Agent] DEBUG: Transcript {i+1} length: {len(transcript) if transcript else 0}")
                
                # Truncate long transcripts for the prompt
                if len(transcript) > 2000:
                    transcript = transcript[:2000] + "... [truncated]"
                
                transcript_summaries.append({
                    "title": result["title"],
                    "transcript_summary": transcript
                })
            else:
                print(f"[Corporate Meta Writer Agent] WARNING: Transcript {i+1} missing required fields: transcript present: {'transcript' in result}, title present: {'title' in result}")
        
        print(f"[Corporate Meta Writer Agent] DEBUG: Created {len(transcript_summaries)} transcript summaries")
        
        # Prepare RAG results
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Preparing RAG results")
        rag_findings = []
        
        # Validate rag_results
        if rag_results is None:
            print(f"[Corporate Meta Writer Agent] WARNING: rag_results is None, initializing as empty dict")
            rag_results = {}
            
        for query, results in rag_results.items():
            print(f"[Corporate Meta Writer Agent] DEBUG: Processing RAG query: {query}")
            
            # Check if results is None
            if results is None:
                print(f"[Corporate Meta Writer Agent] WARNING: Results for query '{query}' is None")
                continue
                
            print(f"[Corporate Meta Writer Agent] DEBUG: Query has {len(results)} results")
            
            relevant_texts = []
            # Take top 3 results per query
            for i, result in enumerate(results[:3] if results else []):  
                if result is None:
                    print(f"[Corporate Meta Writer Agent] WARNING: Result {i+1} for query '{query}' is None")
                    continue
                    
                text = result.get("text", "")
                if text is None:
                    print(f"[Corporate Meta Writer Agent] WARNING: Text for result {i+1} is None")
                    text = ""
                    
                # Truncate long texts
                relevant_texts.append(text[:500])
            
            rag_findings.append({
                "query": query,
                "findings": relevant_texts
            })
        
        print(f"[Corporate Meta Writer Agent] DEBUG: Created {len(rag_findings)} RAG findings")
        
        # Summarize corporate results
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Summarizing corporate results")
        corporate_summary = {}
        
        # Validate corporate_results
        if corporate_results is None:
            print(f"[Corporate Meta Writer Agent] WARNING: corporate_results is None, initializing as empty dict")
            corporate_results = {}
            
        for data_type, data in corporate_results.items():
            print(f"[Corporate Meta Writer Agent] DEBUG: Processing corporate data type: {data_type}")
            
            # Check if data is None
            if data is None:
                print(f"[Corporate Meta Writer Agent] WARNING: Data for type '{data_type}' is None")
                corporate_summary[data_type] = "No data available"
                continue
                
            if isinstance(data, list):
                print(f"[Corporate Meta Writer Agent] DEBUG: Data type '{data_type}' is a list with {len(data)} items")
                if data:
                    corporate_summary[data_type] = f"{len(data)} entries found"
                else:
                    corporate_summary[data_type] = "No entries found"
            elif isinstance(data, dict):
                print(f"[Corporate Meta Writer Agent] DEBUG: Data type '{data_type}' is a dict with {len(data)} items")
                if data:
                    corporate_summary[data_type] = f"{len(data)} entries found"
                else:
                    corporate_summary[data_type] = "Empty dictionary"
            else:
                print(f"[Corporate Meta Writer Agent] WARNING: Data type '{data_type}' has unexpected type: {type(data)}")
                corporate_summary[data_type] = "No data available"
        
        print(f"[Corporate Meta Writer Agent] DEBUG: Created corporate summary with {len(corporate_summary)} entries")
        
        # Generate the report
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Preparing prompt for LLM")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Validate input for JSON serialization
        print(f"[Corporate Meta Writer Agent] DEBUG: Validating data for JSON serialization")
        try:
            json.dumps(rag_findings, indent=2)
            print(f"[Corporate Meta Writer Agent] DEBUG: RAG findings successfully serialized to JSON")
        except Exception as e:
            print(f"[Corporate Meta Writer Agent] ERROR: Failed to serialize RAG findings to JSON: {str(e)}")
            rag_findings = [{"query": "Error in data", "findings": ["Error serializing RAG data"]}]
            
        try:
            json.dumps(transcript_summaries, indent=2)
            print(f"[Corporate Meta Writer Agent] DEBUG: Transcript summaries successfully serialized to JSON")
        except Exception as e:
            print(f"[Corporate Meta Writer Agent] ERROR: Failed to serialize transcript summaries to JSON: {str(e)}")
            transcript_summaries = [{"title": "Error", "transcript_summary": "Error serializing transcript data"}]
            
        try:
            json.dumps(corporate_summary, indent=2)
            print(f"[Corporate Meta Writer Agent] DEBUG: Corporate summary successfully serialized to JSON")
        except Exception as e:
            print(f"[Corporate Meta Writer Agent] ERROR: Failed to serialize corporate summary to JSON: {str(e)}")
            corporate_summary = {"Error": "Failed to serialize corporate data"}
        
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
        2. Key Personel
        2. Business Overview
        3. Review Of Document Analysis Results 
        4. Major Announcements made over the last Year 
        5. Summary of last 4 Conference Calls
        6. Major Governance Concerns
        
        
        Key personel should have the details of board memebers,various comitees and their members. (NSE Corporate data)
        Review Of Document Analysis Results should contain a comprehensive detail based on document analysis result which retrieves information from the ESG Report of the company
        Major Announcemenrs made over the last year should contain the 10 most important announcements and details of it. (NSE Corporate Data)
        Summary of last 4 conference calls should have a section elabborating each conference call and then an overall summary. 

        Format the report in Markdown. Make it detailed, insightful, and professional.
        """
        
        print(f"[Corporate Meta Writer Agent] STEP: Generate Final Report - Constructing messages for LLM")
        messages = [
            ("system", "You are an expert financial analyst specializing in comprehensive corporate research."),
            ("human", prompt)
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
    
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"[Corporate Meta Writer Agent] ERROR: PDF file not found at {pdf_path}")
        return {**state, "goto": "END", "error": f"PDF file not found at {pdf_path}"}
    
    # Initialize LLM
    print("[Corporate Meta Writer Agent] MAIN: Initializing LLM")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        print("[Corporate Meta Writer Agent] DEBUG: LLM initialized successfully")
    except Exception as e:
        print(f"[Corporate Meta Writer Agent] ERROR: Failed to initialize LLM: {e}")
        return {**state, "goto": "END", "error": f"Failed to initialize LLM: {str(e)}"}
    
    # Track workflow state
    current_step = state.get("corporate_meta_step", "start")
    print(f"[Corporate Meta Writer Agent] MAIN: Current step: '{current_step}'")
    
    # STEP 1: Initial setup and route to RAG agent
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
        
        # Update state for RAG agent
        state["rag_queries"] = queries
        state["rag_pdf_path"] = pdf_path
        state["is_file_embedded"] = False
        state["corporate_meta_step"] = "post_rag"
        
        print(f"[Corporate Meta Writer Agent] STEP 1: Routing to rag_agent with {len(queries)} queries")
        return {**state, "goto": "rag_agent"}
    
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