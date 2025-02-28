import json
import os
from typing import Dict, List, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def load_preliminary_research_guidelines(company: str, industry: str) -> List[str]:
    """
    Load preliminary research guidelines from file or generate default ones if file doesn't exist.
    """
    try:
        with open("preliminary_research.json", "r") as file:
            guidelines = json.load(file)
            return guidelines
    except Exception as e:
        raise Exception("No preliminary research guidelines found")
        

def evaluate_research_quality(company: str, industry: str, research_results: Dict) -> Dict:
    """
    Evaluate the quality, completeness, and balance of the research results.
    Returns a quality assessment with scores and recommendations.
    """
    if not research_results:
        return {
            "overall_score": 0,
            "coverage_score": 0,
            "balance_score": 0,
            "credibility_score": 0,
            "assessment": "No research results available.",
            "recommendations": {"default recommendation": "Continue with available research while addressing technical issues."}
        }
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    system_prompt = (
        "You are an expert in research quality assessment and corporate intelligence analysis. Your task is to evaluate the provided research results "
        "to determine their completeness, credibility, and effectiveness in identifying risks and concerns related to a company. The assessment should focus on the following key factors:\n\n"
        "1. COVERAGE – Does the research thoroughly address all major areas, including operational, legal, management, and reputation? Are there any critical gaps?\n"
        "2. CREDIBILITY – Are the sources authoritative, well-substantiated, and free from unreliable or speculative claims?\n"
        "3. BALANCE (RISK-ORIENTED) – Does the research provide a mix of perspectives while prioritizing negative aspects, risks, and red flags over neutral or positive information?\n\n"
        "Key Considerations:\n"
        "- Coverage and credibility are the most important factors.\n"
        "- Balance should be skewed towards identifying risks and concerns rather than presenting a neutral or positive picture.\n"
        "- Take into account the specific industry or company context. If the industry is provided, adjust the scoring accordingly; if not, infer the industry from the information. For example, if the industry is one where fraud is uncommon, be more considerate in rating fraud-related risks.\n\n"
        "After evaluating the research quality, identify specific gaps that should be addressed for a comprehensive analysis of the company. Consider these potential areas:\n\n"
        "1. Regulatory compliance record\n"
        "2. Legal issues and litigation history\n"
        "3. Corporate governance practices\n"
        "4. Competitive position in the industry\n"
        "5. Reputation and customer satisfaction\n"
        "6. Environmental, social, and governance (ESG) factors\n"
        "7. Innovation and R&D pipeline\n"
        "8. International operations and risks\n\n"
        "Return a JSON object with the following fields:\n\n"
        "{\n"
        '  "overall_score": 0-10,\n'
        '  "coverage_score": 0-10,\n'
        '  "credibility_score": 0-10,\n'
        '  "balance_score": 0-10,\n'
        '  "assessment": "Brief summary of the research quality",\n'
        '  "recommendations": {\n'
        '      "<area_1>": "Key aspect missing and what should be explored",\n'
        '      "<area_2>": "Key aspect missing and what should be explored",\n'
        '      "<area_3>": "Key aspect missing and what should be explored"\n'
        '  }\n'
        "}\n\n"
        "The 'recommendations' field should contain up to three areas where further research is needed. If there are fewer than three gaps, include only the relevant ones."
    )

    input_message = (
        f"Company: {company}\n",
        f"Industry: {industry}\n"
        f"Research Results: {json.dumps(research_results)}\n\n"
        f"Evaluate the quality of these research results for {company} and identify the most important research gaps."
    )
    
    messages = [
        ("system", system_prompt),
        ("human", input_message)
    ]
    
    try:
        response = llm.invoke(messages)
        assessment = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        return assessment
    except Exception as e:
        print(f"[Meta Agent] Error in research quality evaluation: {e}")
        return {
            "overall_score": 5,  
            "coverage_score": 5,
            "balance_score": 5,
            "credibility_score": 5,
            "assessment": "Unable to evaluate research quality due to an error.",
            "recommendations": {"default recommendation": "Continue with available research while addressing technical issues."}
        }

def identify_research_gaps(company: str, industry: str, event_name: str, 
                           event_data: List[Dict], previous_research_plans: List[Dict]) -> Dict[str, str]:
    """
    Evaluate the completeness of an event's research and analysis.
    Identify if there are gaps in the story, missing facts, or areas requiring further research.
    
    Args:
        company: The company being analyzed
        industry: The industry of the company
        event_name: The name of the event being analyzed
        event_data: The collected insights about the event
        previous_research_plans: Previously executed research plans
        
    Returns:
        A dictionary of gap topics and descriptions, or empty dict if no gaps
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
        system_prompt = (
            "You are an expert corporate intelligence analyst. Your task is to assess whether the provided event "
            "and its analysis present a complete and well-supported account of the situation. You should determine:\n\n"
            "1. COMPLETENESS – Does the event description and analysis provide all essential details? Are key elements "
            "such as the entity responsible, the individuals involved, and critical context fully covered?\n"
            "2. FACT GAPS – Are there any missing or unclear facts that are necessary to understand the event properly? "
            "If so, identify them.\n"
            "3. REDUNDANCY CHECK – If a fact gap exists, check whether it has already been addressed in previous research "
            "plans. If it has, DO NOT request it again.\n\n"
            "### RESPONSE FORMAT\n"
            "- If the event and analysis are complete, return an empty JSON object: {}\n"
            "- If gaps exist, return a dictionary where:\n"
            "  - Keys represent the missing topics (e.g., 'Entity Responsible', 'Key Witness', 'Financial Impact').\n"
            "  - Values provide a brief description of the missing details.\n\n"
            "Be concise and only request information IF it is truly necessary to complete the story."
        )
        
        # Extract previous queries to check for redundancy
        previous_queries = []
        for plan in previous_research_plans:
            query_cats = plan.get("query_categories", {})
            for cat, desc in query_cats.items():
                previous_queries.append(f"{cat}: {desc}")
        
        input_message = (
            f"Company: {company}\n"
            f"Industry: {industry}\n"
            f"Event: {event_name}\n"
            f"Event Analysis: {json.dumps(event_data)}\n"
            f"Previous Research Plans: {json.dumps(previous_queries)}\n"
            f"\nAssess whether the event is fully understood and identify any necessary research gaps."
        )
        
        messages = [
            ("system", system_prompt),
            ("human", input_message)
        ]
        
        response = llm.invoke(messages)
        response_content = response.content.strip()
        
        # Extract JSON content
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content
            
        gaps = json.loads(json_content)
        
        # Validate and return
        if isinstance(gaps, dict):
            if gaps:
                print(f"[Meta Agent] Found {len(gaps)} research gaps for event '{event_name}'")
                for topic, description in gaps.items():
                    print(f"  - {topic}: {description}")
            else:
                print(f"[Meta Agent] No research gaps found for event '{event_name}'")
            return gaps
        else:
            print(f"[Meta Agent] Invalid response format for event '{event_name}', expected dict but got {type(gaps)}")
            return {}
            
    except Exception as e:
        print(f"[Meta Agent] Error in identifying research gaps: {e}")
        import traceback
        print(traceback.format_exc())
        return {}

def create_research_plan(company: str, research_gaps: Union[List[Dict], Dict], previous_plans: List[Dict] = None) -> Dict:
    """
    Create a structured research plan based on identified research gaps.
    """
    if not research_gaps:
        return {}
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    system_prompt = (
        "You are an expert research strategist specializing in corporate intelligence. "
        "Your goal is to generate a structured research plan that addresses research gaps "
        "while avoiding unnecessary repetition and demonstrating analytical creativity.\n\n"
        
        "### INSTRUCTIONS:\n"
        "1. Review the provided research gaps thoroughly.\n"
        "2. Check previous research plans to avoid redundant efforts.\n"
        "3. If a previous research plan failed, develop alternative approaches rather than repeating the same strategy.\n"
        "4. View research gaps from multiple perspectives and levels of abstraction.\n"
        "5. Apply creative problem-solving when direct data is unavailable:\n"
        "   - Identify proxy metrics or indirect indicators\n"
        "   - Suggest triangulation methods using related accessible data\n"
        "   - Recommend industry benchmarks or comparable situations\n"
        "   - Outline estimation techniques using partial data\n"
        "6. For financial data gaps (like property values, acquisition costs), suggest:\n"
        "   - Finding comparable market rates and scaling based on size/location\n"
        "   - Identifying industry-standard valuation methods\n"
        "   - Leveraging public records, tax assessments, or regulatory filings\n"
        "   - Using statistical methods to establish reasonable ranges\n"
        "7. Always prioritize specificity and actionability in your recommendations.\n\n"
        
        "### RESPONSE FORMAT:\n"
        "You must return a fully structured research plan in the following JSON format:\n"
        "{\n"
        "  \"objective\": \"A concise summary of the research goal.\",\n"
        "  \"key_areas_of_focus\": [\"List of critical topics requiring investigation\"],\n"
        "  \"query_categories\": {\n"
        "    \"category_name\": \"Description of research focus for this category\"\n"
        "  },\n"
        "  \"query_generation_guidelines\": \"Guidelines for structuring search queries\"\n"
        "}\n"
        "Ensure all fields are populated with substantive content. Do NOT return a response with only 'research_gaps'."
    )
    
    input_data = {
        "company": company,
        "research_gaps": research_gaps,
        "previous_plans": previous_plans or []
    }
    
    input_message = f"Create a comprehensive research plan for investigating {company} based on the following gaps and any previous plans:\n{json.dumps(input_data, indent=2)}"
    
    messages = [
        ("system", system_prompt),
        ("human", input_message)
    ]
    
    try:
        response = llm.invoke(messages)
        cleaned_response = response.content.replace("```json", "").replace("```", "").strip()
        plan = json.loads(cleaned_response)
        
        if not all(k in plan for k in ["objective", "key_areas_of_focus", "query_categories", "query_generation_guidelines"]):
            return {
                "objective": "To investigate and address identified research gaps related to " + company,
                "key_areas_of_focus": [gap for gap in research_gaps] if isinstance(research_gaps, list) else ["Investigate " + str(research_gaps)],
                "query_categories": {"general": "Primary investigation focus for identified gaps"},
                "query_generation_guidelines": "Generate specific, targeted queries to address each research gap effectively."
            }
        
        return plan
    except Exception as e:
        print(f"[Meta Agent] Error in processing research plan: {e}")
        return {
            "objective": "To investigate key aspects of " + company,
            "key_areas_of_focus": ["Address identified information gaps"],
            "query_categories": {"primary": "Focus on core research needs"},
            "query_generation_guidelines": "Create targeted search queries to find relevant information."
        }

def generate_analysis_guidance(company: str, research_results: Dict) -> Dict:
    """
    Generate guidance for the analyst agent based on research results.
    Returns structured guidance with focus areas and analysis strategies.
    """
    if not research_results:
        return {
            "focus_areas": ["General company assessment"],
            "priorities": ["Establish baseline understanding of company"],
            "analysis_strategies": ["Conduct general background research"],
            "red_flags": ["Insufficient information available"]
        }
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    system_prompt = (
        "You are an expert in corporate forensics and due diligence. Based on the research results, "
        "provide structured guidance for the analysis phase. Identify the most important areas to "
        "focus on, potential red flags that require deeper investigation, and effective analytical "
        "strategies for this specific company.\n\n"
        "Return a JSON object with:\n"
        "- 'focus_areas': List of specific topics that deserve primary attention\n"
        "- 'priorities': List of events/issues ranked by importance for analysis\n"
        "- 'analysis_strategies': List of specific approaches to extract maximum insights\n"
        "- 'red_flags': List of concerning patterns or issues requiring deeper investigation\n"
        "- 'context_recommendations': List of additional context needed for proper analysis"
    )
    
    input_message = (
        f"Company: {company}\n"
        f"Research Results: {json.dumps(research_results)}\n\n"
        f"Generate analysis guidance for {company} based on these research results."
    )
    
    messages = [
        ("system", system_prompt),
        ("human", input_message)
    ]
    
    try:
        response = llm.invoke(messages)
        guidance = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        return guidance
    except Exception as e:
        print(f"[Meta Agent] Error in generating analysis guidance: {e}")
        return {
            "focus_areas": ["Overview of all identified events"],
            "priorities": ["Most recent events", "Events with highest impact ratings"],
            "analysis_strategies": ["Compare events chronologically", "Look for patterns in company behavior"],
            "red_flags": ["Any recurring issues", "Any regulatory actions"],
            "context_recommendations": ["Industry context", "Company history"]
        }

def evaluate_analysis_completeness(company: str, industry: str, research_results: Dict, analysis_results: Dict) -> Dict:
    """
    Evaluate the completeness of the analysis and identify areas that need more research.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    system_prompt = (
        "You are an expert in corporate due diligence and risk assessment. Evaluate the completeness "
        "of the analysis based on the research results. Identify any areas where the analysis might be "
        "incomplete due to insufficient research data.\n\n"
        "Return a JSON object with:\n"
        "- 'completeness_score': A score from 0-10 indicating how complete the analysis is\n"
        "- 'missing_aspects': List of aspects that couldn't be properly analyzed due to data gaps\n"
        "- 'research_recommendations': List of specific research areas to fill critical gaps\n"
        "- 'confidence_assessment': Brief assessment of overall confidence in the analysis"
    )
    
    input_message = (
        f"Company: {company}\n"
        f"Industry: {industry}\n"
        f"Research Results: {json.dumps(research_results)}\n"
        f"Analysis Results: {json.dumps(analysis_results)}\n\n"
        f"Evaluate the completeness of the analysis for {company} and identify research gaps."
    )
    
    messages = [
        ("system", system_prompt),
        ("human", input_message)
    ]
    
    try:
        response = llm.invoke(messages)
        evaluation = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        return evaluation
    except Exception as e:
        print(f"[Meta Agent] Error in evaluating analysis completeness: {e}")
        return {
            "completeness_score": 5,
            "missing_aspects": ["Comprehensive evaluation not possible due to technical error"],
            "research_recommendations": ["Continue with available information"],
            "confidence_assessment": "Unable to fully evaluate analysis completeness due to an error."
        }

def meta_agent(state: Dict) -> Dict:
    """
    Intelligent orchestration agent that evaluates research quality,
    identifies gaps, creates research plans, and provides guidance.
    Supports multiple iterations between research and analysis.
    """
    print("[Meta Agent] Received state:", state)
    
    # Initialize tracking variables if not present
    if "meta_iteration" not in state:
        state["meta_iteration"] = 0
    if "search_history" not in state:
        state["search_history"] = []
    if "event_research_iterations" not in state:
        state["event_research_iterations"] = {}
    
    # Set limits
    research_threshold = 6  # Minimum quality score to proceed to analysis
    max_iterations = 3      # Maximum overall iterations
    max_event_iterations = 2  # Maximum iterations per event
    
    # Increment iteration counter
    state["meta_iteration"] += 1
    current_iteration = state["meta_iteration"]
    print(f"[Meta Agent] Starting iteration {current_iteration}/{max_iterations}")
    
    # Extract key information from state
    company = state.get("company", "")
    industry = state.get("industry", "Unknown")
    research_results = state.get("research_results", {})
    analysis_results = state.get("analysis_results", {})
    previous_research_plans = state.get("research_plan", [])
    
    # Validation - Company name is required
    if not company:
        print("[Meta Agent] ERROR: 'company' key is missing!")
        return {**state, "goto": "END", "error": "Company name is missing"}
    
    # STEP 1: Initial Research Phase - Load preliminary guidelines and start research
    if not research_results and current_iteration == 1:
        print("[Meta Agent] Starting preliminary research...")
        try:
            preliminary_guidelines = load_preliminary_research_guidelines(company, industry)
            state["research_plan"] = [preliminary_guidelines]
            state["search_type"] = "google_news"
            state["return_type"] = "clustered"
            print(f"[Meta Agent] Loaded preliminary research guidelines")
            return {**state, "goto": "research_agent"}
        except Exception as e:
            print(f"[Meta Agent] Error loading preliminary guidelines: {e}")
            # Fallback to basic research
            basic_plan = {
                "objective": f"Investigate potential issues related to {company}",
                "key_areas_of_focus": ["Legal issues", "Financial concerns", "Regulatory actions"],
                "query_categories": {"general": "Investigate potential issues"},
                "query_generation_guidelines": "Focus on negative news and regulatory concerns"
            }
            state["research_plan"] = [basic_plan]
            state["search_type"] = "google_news"
            state["return_type"] = "clustered"
            return {**state, "goto": "research_agent"}
    
    # STEP 2: Research Quality Evaluation
    if research_results and (not state.get("quality_assessment") or 
                            state.get("quality_assessment", {}).get("overall_score", 0) < research_threshold):
        print("[Meta Agent] Evaluating research quality...")
        quality_assessment = evaluate_research_quality(company, industry, research_results)
        state["quality_assessment"] = quality_assessment
        
        print(f"[Meta Agent] Research quality score: {quality_assessment.get('overall_score', 0)}/10")
        print(f"[Meta Agent] Assessment: {quality_assessment.get('assessment', 'N/A')}")
        
        # Check if additional research is needed
        if quality_assessment.get('overall_score', 0) < research_threshold and current_iteration < max_iterations:
            print("[Meta Agent] Research quality below threshold. Generating targeted research plan...")
            # Generate targeted research plan based on quality assessment recommendations
            research_gaps = quality_assessment.get('recommendations', {})
            if research_gaps:
                research_plan = create_research_plan(company, research_gaps, previous_research_plans)
                state["research_plan"].append(research_plan)
                print(f"[Meta Agent] Generated new research plan: {research_plan.get('objective', 'No objective')}")
                return {**state, "goto": "research_agent"}
    
    # STEP 3: Move to Analysis when research is sufficient
    if (state.get("quality_assessment", {}).get("overall_score", 0) >= research_threshold or 
        current_iteration >= max_iterations) and not analysis_results:
        print("[Meta Agent] Research quality sufficient or max iterations reached. Moving to analysis phase...")
        # Generate analysis guidance for the analyst agent
        analysis_guidance = generate_analysis_guidance(company, research_results)
        state["analysis_guidance"] = analysis_guidance
        print(f"[Meta Agent] Generated analysis guidance with {len(analysis_guidance.get('focus_areas', []))} focus areas")
        return {**state, "goto": "analyst_agent"}
    
    # STEP 4: Post-Analysis Research Gap Identification (after analyst agent has run)
    if analysis_results and current_iteration < max_iterations:
        print("[Meta Agent] Analysis completed. Identifying research gaps in events...")
        
        # Get event-specific research gaps
        all_event_gaps = {}
        for event_name, event_data in analysis_results.get("forensic_insights", {}).items():
            # Skip events that have already reached max research iterations
            current_event_iterations = state["event_research_iterations"].get(event_name, 0)
            if current_event_iterations >= max_event_iterations:
                print(f"[Meta Agent] Skipping event '{event_name}' - reached max iterations ({max_event_iterations})")
                continue
                
            # Identify gaps for this specific event
            event_gaps = identify_research_gaps(
                company, 
                industry,
                event_name, 
                event_data, 
                previous_research_plans
            )
            
            if event_gaps:
                all_event_gaps[event_name] = event_gaps
                print(f"[Meta Agent] Identified {len(event_gaps)} research gaps for event: {event_name}")
        
        # If we have gaps, create targeted research plans
        if all_event_gaps:
            targeted_plans = []
            for event_name, gaps in all_event_gaps.items():
                event_plan = create_research_plan(
                    company, 
                    gaps, 
                    previous_research_plans
                )
                
                # Track research iterations for this event
                if event_name not in state["event_research_iterations"]:
                    state["event_research_iterations"][event_name] = 0
                state["event_research_iterations"][event_name] += 1
                
                # Store the event name in the plan for tracking
                event_plan["event_name"] = event_name
                targeted_plans.append(event_plan)
            
            if targeted_plans:
                state["research_plan"].extend(targeted_plans)
                print(f"[Meta Agent] Created {len(targeted_plans)} targeted research plans for events with gaps")
                return {**state, "goto": "research_agent"}
    
    # STEP 5: Final Analysis - Run one more analysis pass if we've done additional research
    if analysis_results and state.get("additional_research_completed") and not state.get("final_analysis_completed"):
        print("[Meta Agent] Additional research completed. Running final analysis...")
        state["final_analysis_requested"] = True
        state["final_analysis_completed"] = True
        return {**state, "goto": "analyst_agent"}
    
    # STEP 6: Complete the process - move to final report generation
    print(f"[Meta Agent] Process complete after {current_iteration} iterations")
    final_quality = state.get("quality_assessment", {}).get("overall_score", 0)
    print(f"[Meta Agent] Final research quality: {final_quality}/10")
    
    # Move to final report generation
    return {**state, "goto": "meta_agent_final", "status": "complete"}
