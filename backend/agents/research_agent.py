import json
import time
import re
from typing import Dict, List, Tuple, Set, Optional
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import traceback
import os
import logging
from dotenv import load_dotenv
from backend.utils.prompt_manager import PromptManager

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Use environment variable or default path for prompts directory
prompts_dir = os.environ.get("PROMPTS_DIR", None)
prompt_manager = PromptManager(prompts_dir)

def is_quarterly_report_article(title: str, snippet: str = "") -> bool:
    """
    Determine if an article is about a quarterly or annual financial report.
    """
    title_lower = title.lower()
    snippet_lower = snippet.lower() if snippet else ""
    
    report_terms = [
        'quarterly report', 'q1 report', 'q2 report', 'q3 report', 'q4 report',
        'quarterly results', 'q1 results', 'q2 results', 'q3 results', 'q4 results',
        'quarterly earnings', 'annual report', 'annual results', 'financial results',
        'earnings report', 'quarterly financial', 'year-end results'
    ]
    
    for term in report_terms:
        if term in title_lower or term in snippet_lower:
            return True
    
    if (re.search(r'q[1-4]\s*20[0-9]{2}', title_lower) or 
        re.search(r'fy\s*20[0-9]{2}', title_lower) or
        re.search(r'q[1-4]\s*20[0-9]{2}', snippet_lower) or
        re.search(r'fy\s*20[0-9]{2}', snippet_lower)):
        return True
    
    if (re.search(r'report[s]?\s+\d+%', title_lower) or 
        re.search(r'revenue\s+of\s+[\$£€]', title_lower) or
        re.search(r'profit\s+of\s+[\$£€]', title_lower)):
        return True
    
    return False

def parse_serp_results(raw_output, category: str) -> List[Dict]:
    """
    Enhanced parser that handles both string and list responses from SerpAPI.
    """
    results = []
    
    try:
        if isinstance(raw_output, str):
            logger.debug(f"Attempting to parse string response as JSON")
            try:
                parsed_data = json.loads(raw_output)
                
                if isinstance(parsed_data, list):
                    raw_output = parsed_data
                    logger.debug(f"Successfully parsed string as JSON list with {len(raw_output)} items")
                elif isinstance(parsed_data, dict) and 'organic_results' in parsed_data:
                    raw_output = parsed_data['organic_results']
                    logger.debug(f"Successfully parsed string as JSON object with organic_results")
            except json.JSONDecodeError:
                logger.debug(f"Not valid JSON, treating as text: {raw_output[:100]}...")
                
                if len(raw_output) > 50:
                    import hashlib
                    hash_id = hashlib.md5(raw_output.encode()).hexdigest()[:10]
                    
                    results.append({
                        "index": 0,
                        "title": raw_output[:100] + "..." if len(raw_output) > 100 else raw_output,
                        "link": f"https://placeholder.com/text_{hash_id}",
                        "date": "Unknown date",
                        "snippet": raw_output,
                        "source": "API text response",
                        "category": category,
                        "is_quarterly_report": False
                    })
                    
                    logger.debug(f"Created result from text response")
                    return results
        
        if isinstance(raw_output, list):
            for i, item in enumerate(raw_output):
                if isinstance(item, dict) and "title" in item and "link" in item:
                    # Check if article is about quarterly reports
                    is_quarterly = is_quarterly_report_article(
                        item["title"], 
                        item.get("snippet", "")
                    )
                    
                    results.append({
                        "index": i,
                        "title": item["title"].strip(),
                        "link": item["link"].strip(),
                        "date": item.get("date", "Unknown date").strip(),
                        "snippet": item.get("snippet", "").strip(),
                        "source": item.get("source", "Unknown source").strip(),
                        "category": category,
                        "is_quarterly_report": is_quarterly
                    })
        
    except Exception as e:
        logger.error(f"Error in parse_serp_results: {e}")
        logger.error(traceback.format_exc())
    
    logger.info(f"Parsed {len(results)} results from category '{category}'")
    return results

def calculate_event_importance(event_name: str, articles: List[Dict]) -> int:
    """
    Calculate an importance score for an event based on content indicators.
    Higher numbers indicate more important events.
    """
    score = 50
    
    event_name_lower = event_name.lower()
    
    if any(term in event_name_lower for term in ['quarterly report', 'financial results', 'earnings report', 'agm', 'board meeting']):
        score -= 50
    
    if any(term in event_name_lower for term in ['fraud', 'scam', 'lawsuit', 'investigation', 'scandal', 'fine', 'penalty', 'cbI raid', 'ed probe', 'bribery', 'allegation']):
        score += 30
    
    if 'criminal' in event_name_lower or 'money laundering' in event_name_lower:
        score += 40
    
    if 'class action' in event_name_lower or 'public interest litigation' in event_name_lower:
        score += 25
    
    if any(term in event_name_lower for term in ['sebi', 'rbi', 'cbi', 'ed', 'income tax', 'competition commission']):
        score += 20
    
    article_count = len(articles)
    score += min(article_count * 2.5, 25)  
    
    # Retain the "High" and "Medium" importance factors
    if '- High' in event_name:
        score += 25
    elif '- Medium' in event_name:
        score += 10
    
    reputable_sources = ['economic times', 'business standard', 'mint', 'hindu business line', 'moneycontrol', 'ndtv', 'the hindu', 'times of india']
    for article in articles:
        source = article.get('source', '').lower()
        if any(rep_source in source for rep_source in reputable_sources):
            score += 2
    
    return score

def group_results(company: str, articles: List[Dict], industry: str = None) -> Dict[str, Dict[str, any]]:
    """
    Group news articles into event clusters with special handling for quarterly reports
    and improved prioritization of negative events.
    """
    if not articles:
        logger.warning("No articles to cluster")
        return {}
    
    # Split articles into quarterly reports and others
    quarterly_report_articles = [a for a in articles if a.get("is_quarterly_report", False)]
    other_articles = [a for a in articles if not a.get("is_quarterly_report", False)]
    
    logger.info(f"Identified {len(quarterly_report_articles)} quarterly report articles")
    logger.info(f"Processing {len(other_articles)} non-quarterly report articles")
    
    regular_events = {}
    if other_articles:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

            simplified_articles = []
            for i, article in enumerate(other_articles):
                simplified_articles.append({
                    "index": i,
                    "title": article["title"],
                    "snippet": article.get("snippet", ""),
                    "date": article.get("date", "Unknown date"),
                    "source": article.get("source", "Unknown source"),
                    "category": article.get("category", "general")
                })

            variables = {
                "company": company,
                "industry": industry,
                "simplified_articles": json.dumps(simplified_articles)
            }

            try:
                system_prompt, human_prompt = prompt_manager.get_prompt("research_agent", "group_results", variables)
            except Exception as e:
                logger.error(f"Error getting prompt for grouping results: {e}")
                # Fallback prompt
                system_prompt = "You are an expert at organizing financial news into logical event categories."
                human_prompt = f"""
                Group these articles about {company} into distinct event categories:
                {json.dumps(simplified_articles, indent=2)}
                
                Return a JSON dictionary where:
                - Keys are event names with format "Event: Brief description - Priority"
                - Values are lists of article indices (numbers only) that belong to each event
                
                Example:
                {{
                  "Event: Company Q1 Results - Low": [0, 5, 9],
                  "Event: CEO Resignation - High": [2, 3, 7]
                }}
                """
        
            messages = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]

            response = llm.invoke(messages)
            response_content = response.content.strip()
            
            # Extract JSON content with better error handling
            try:
                if "```json" in response_content:
                    json_content = response_content.split("```json")[1].split("```")[0].strip()
                elif "```" in response_content:
                    json_content = response_content.split("```")[1].strip()
                else:
                    json_content = response_content
                    
                clustered_indices = json.loads(json_content)
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse LLM grouping response as JSON: {e}")
                # Create a simple grouping as fallback
                clustered_indices = {}
                for i, article in enumerate(other_articles):
                    event_name = f"News: {article['title'][:50]}..."
                    clustered_indices[event_name] = [i]
            
            for event_name, indices in clustered_indices.items():
                valid_indices = []
                for idx in indices:
                    if isinstance(idx, str) and idx.isdigit():
                        idx = int(idx)
                    if isinstance(idx, int) and 0 <= idx < len(other_articles):
                        valid_indices.append(idx)
                
                if valid_indices:
                    regular_events[event_name] = [other_articles[i] for i in valid_indices]
            
            logger.info(f"Grouped non-quarterly articles into {len(regular_events)} events")
            
        except Exception as e:
            logger.error(f"Error clustering non-quarterly articles: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback: create one event per article
            for i, article in enumerate(other_articles):
                event_name = f"News: {article['title'][:50]}..."
                regular_events[event_name] = [article]
            
            logger.info(f"Created {len(regular_events)} individual article events as fallback")
    
    # Handle quarterly report articles
    if quarterly_report_articles:
        dates = [article.get("date", "") for article in quarterly_report_articles]
        valid_dates = [d for d in dates if d and d.lower() != "unknown date"]
        
        date_str = ""
        if valid_dates:
            try:
                parsed_dates = []
                for date_text in valid_dates:
                    try:
                        for fmt in ["%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y"]:
                            try:
                                parsed_date = datetime.strptime(date_text, fmt)
                                parsed_dates.append(parsed_date)
                                break
                            except:
                                continue
                    except:
                        pass
                
                if parsed_dates:
                    most_recent = max(parsed_dates)
                    date_str = f" ({most_recent.strftime('%b %Y')})"
            except Exception as e:
                logger.error(f"Error parsing dates for quarterly reports: {e}")
                date_str = f" ({valid_dates[0]})" if valid_dates else ""
        
        quarterly_event_name = f"Financial Reporting: Quarterly/Annual Results{date_str} - Low"
        regular_events[quarterly_event_name] = quarterly_report_articles
        logger.info(f"Created a consolidated event for {len(quarterly_report_articles)} quarterly report articles")
    
    # Calculate importance scores and build final events structure
    final_events = {}
    importance_scores = {}
    
    for event_name, event_articles in regular_events.items():
        importance = calculate_event_importance(event_name, event_articles)
        importance_scores[event_name] = importance
        
        event_data = {
            "articles": event_articles,
            "importance_score": importance,
            "article_count": len(event_articles)
        }
        final_events[event_name] = event_data
    
    logger.info(f"Assigned importance scores to {len(importance_scores)} events")
    for event, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"Event: '{event}' - Score: {score}")
    
    return final_events

def generate_queries(company: str, industry: str, research_plan: Dict, query_history: List[Dict[str, List]]) -> Dict[str, List[str]]:
    """
    Generate search queries based on a research plan given by the meta agent, research plans can be basic research guidelines or hyper specific questions.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

        variables = {
            "company": company,
            "industry": industry,
            "research_plan": json.dumps(research_plan, indent=4),
            "query_history": json.dumps(query_history, indent=4), 
        }
        
        try:
            system_prompt, human_prompt = prompt_manager.get_prompt("research_agent", "generate_queries", variables)
            logger.info(f"Successfully loaded prompts for query generation")
        except Exception as e:
            logger.error(f"Error getting prompt for query generation: {e}")
            # Fallback to hardcoded prompt
            system_prompt = "You are an expert search query generator focused on investigating companies."
            human_prompt = f"""
            Generate search queries to investigate {company} in the {industry} industry.
            
            Research plan: {json.dumps(research_plan, indent=4)}
            
            Generate a JSON object with categories as keys and arrays of search queries as values.
            Example format: {{"Regulatory Issues": ["{company} regulatory violations", "{company} regulatory investigation"]}}
            """
            logger.info("Using fallback prompt for query generation")

        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        logger.debug(f"Invoking LLM for query generation")
        response = llm.invoke(messages)
        response_content = response.content.strip()
        logger.debug(f"Received response from LLM for query generation")
        
        # Extract JSON content with better error handling
        try:
            if "```json" in response_content:
                json_content = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_content = response_content.split("```")[1].strip()
            else:
                json_content = response_content
                
            query = json.loads(json_content)
            logger.info(f"Successfully parsed LLM response as JSON with {len(query)} categories")
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response content: {response_content[:200]}...")
            # Create fallback queries
            query = {
                "General Concerns": [
                    f"{company} financial irregularities",
                    f"{company} regulatory issues",
                    f"{company} legal problems",
                    f"{company} controversy"
                ]
            }
            logger.info("Using fallback queries due to JSON parsing error")
            
        query_count = sum(len(v) for v in query.values())
        logger.info(f"Generated {query_count} queries across {len(query)} categories")
        return query

    except Exception as e:
        logger.error(f"Error in query generation: {e}")
        logger.error(traceback.format_exc())
        # Return fallback queries when exceptions occur
        return {
            "General": [
                f"{company} issues",
                f"{company} controversy",
                f"{company} regulatory problems",
                f"{company} financial concerns"
            ]
        }


def research_agent(state: Dict) -> Dict:
    """
    Enhanced research agent that handles both preliminary and targeted research.
    Prevents duplicate queries and efficiently processes search results.
    """
    logger.info(f"Research Agent received state for company: {state.get('company', 'Unknown')}")
    
    # Extract key information from state
    company = state.get("company", "")
    industry = state.get("industry", "Unknown")
    research_plans = state.get("research_plan", [])
    search_history = state.get("search_history", [])
    current_results = state.get("research_results", {})
    search_type = state.get("search_type", "google_search")
    return_type = state.get("return_type", "clustered")
    
    # Validation - Company name and research plan are required
    if not company:
        logger.error("Missing company name")
        return {**state, "goto": "meta_agent", "error": "Missing company name"}
        
    if not research_plans:
        logger.error("No research plan provided")
        return {**state, "goto": "meta_agent", "error": "No research plan provided"}
    
    # Get the most recent research plan
    current_plan = research_plans[-1]
    logger.info(f"Processing research plan: {current_plan.get('objective', 'No objective specified')}")
    
    # Check if this is an event-specific targeted research plan
    target_event = current_plan.get("event_name", None)
    if target_event:
        logger.info(f"This is a targeted research plan for event: {target_event}")
    
    # Set up output tracking
    all_articles = []
    executed_queries = []
    
    # Generate search queries based on the research plan
    try:
        query_categories = generate_queries(company, industry, current_plan, search_history)
        
        # Limit number of queries per category for efficiency
        for category, queries in query_categories.items():
            if len(queries) > 3:
                logger.info(f"Limiting {category} from {len(queries)} to 3 queries")
                query_categories[category] = queries[:3]
        
        # Configure SerpAPI parameters with environment check
        serp_api_key = os.environ.get("SERPAPI_API_KEY")
        if not serp_api_key:
            logger.warning("SERPAPI_API_KEY environment variable not found")
        
        # Execute searches and collect results
        for category, queries in query_categories.items():
            logger.info(f"Processing category: {category}")
            for query in queries:
                # Skip duplicate queries that have been executed before
                if any(query in sublist for sublist in search_history if sublist):
                    logger.info(f"Skipping duplicate query: {query}")
                    continue
                
                logger.info(f"Executing search query: {query}")
                executed_queries.append(query)
                
                # Set up the search parameters
                params = {
                    "engine": "google",
                    "q": query,
                    "location": "India",
                    "google_domain": "google.co.in",
                    "gl": "in",
                    "hl": "en",
                    "safe": "off", 
                    "num": "50",  # Reduced from 100 to improve reliability
                    "output": "json"
                }
                if search_type == "google_news":
                    params["tbm"] = "nws"
                
                try:
                    # Execute the search
                    serp = SerpAPIWrapper(params=params)
                    raw_output = serp.run(query)
                    
                    # Process the results
                    articles = parse_serp_results(raw_output, category)
                    all_articles.extend(articles)
                    
                    # Add a delay to avoid rate limiting
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error executing query '{query}': {e}")
                    logger.error(traceback.format_exc())
                
    except Exception as e:
        logger.error(f"Error in search execution: {e}")
        logger.error(traceback.format_exc())
    
    # Update search history with the executed queries
    if executed_queries:
        search_history.append(executed_queries)
    state["search_history"] = search_history
    
    logger.info(f"Collected {len(all_articles)} total articles across all categories")
    
    # Handle empty results with a fallback query
    if not all_articles:
        logger.warning("No articles found with targeted queries. Trying fallback query.")
        try:
            fallback_query = f'"{company}" negative news'
            if not any(fallback_query in sublist for sublist in search_history if sublist):
                params = {
                    "engine": "google",
                    "q": fallback_query,
                    "num": "50",
                    "location": "India",
                    "google_domain": "google.co.in",
                    "gl": "in"
                }
                
                if not search_history:
                    search_history.append([])
                search_history[-1].append(fallback_query)
                
                serp = SerpAPIWrapper(params=params)
                raw_output = serp.run(fallback_query)
                fallback_articles = parse_serp_results(raw_output, "general")
                all_articles.extend(fallback_articles)
                logger.info(f"Fallback query returned {len(fallback_articles)} articles")
        except Exception as e:
            logger.error(f"Error with fallback query: {e}")
            logger.error(traceback.format_exc())
    
    # Deduplicate articles by URL
    unique_articles = []
    seen_urls = set()
    for article in all_articles:
        if article["link"] not in seen_urls:
            seen_urls.add(article["link"])
            unique_articles.append(article)
    
    logger.info(f"Deduplicated to {len(unique_articles)} unique articles")
    
    # Process results based on whether this is targeted research for a specific event
    if target_event and return_type == "clustered":
        # For targeted event research, add new articles to the existing event
        if target_event in current_results:
            existing_articles = current_results[target_event]
            existing_urls = {article["link"] for article in existing_articles}
            
            # Only add new articles that aren't already in this event
            new_articles = [a for a in unique_articles if a["link"] not in existing_urls]
            current_results[target_event].extend(new_articles)
            
            logger.info(f"Added {len(new_articles)} new articles to event: {target_event}")
        else:
            # Create a new event entry if it doesn't exist
            current_results[target_event] = unique_articles
            logger.info(f"Created new event '{target_event}' with {len(unique_articles)} articles")
        
        # Signal to meta_agent that we completed additional research
        state["additional_research_completed"] = True
    else:
        # For general research, cluster results into events
        if return_type == "clustered" and unique_articles:
            # Group articles into events
            grouped_results = group_results(company, unique_articles, industry)
            
            # Extract article lists and metadata
            final_results = {}
            event_metadata = {}
            
            for event_name, event_data in grouped_results.items():
                final_results[event_name] = event_data["articles"]
                event_metadata[event_name] = {
                    "importance_score": event_data["importance_score"],
                    "article_count": event_data["article_count"],
                    "is_quarterly_report": any(a.get("is_quarterly_report", False) for a in event_data["articles"])
                }
            
            # Update state with the grouped results
            state["research_results"] = final_results
            state["event_metadata"] = event_metadata
            logger.info(f"Grouped articles into {len(final_results)} distinct events")
        elif return_type != "clustered":
            # Return unclustered results if requested
            state["research_results"] = unique_articles
            logger.info(f"Returning {len(unique_articles)} unclustered articles")
    
    return {**state, "goto": "meta_agent"}