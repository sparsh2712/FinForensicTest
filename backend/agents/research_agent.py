import json
import time
import re
from typing import Dict, List, Tuple, Set, Optional
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import traceback
import os
from dotenv import load_dotenv
load_dotenv()


def generate_queries(company: str, industry: str, research_plan: Dict, query_history: List[Dict[str, List]]) -> Dict[str, List[str]]:
    """
    Generate search queries based on a research plan given by the meta agent, research plans can be basic research guidelines or hyper specific questions.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

        system_prompt = (
            "You are an advanced research assistant specializing in forensic and investigative analysis. "
            "Your task is to generate precise and diverse search queries based on a given research plan. "
            "These queries should help uncover relevant information about a company while ensuring that redundant or similar queries are not generated.\n\n"

            "### Instructions:\n"
            "1. Read the provided research plan carefully. It contains detailed objectives, key focus areas, and specific categories for which queries should be generated.\n"
            "2. Review the list of previously generated queries to ensure that new queries are distinct and do not repeat past attempts. "
            "Generating queries that are too similar can lead to redundant search results.\n"
            "3. Construct well-formed and effective search queries using appropriate search operators and keywords. "
            "Each query should be structured to maximize relevant results and align with the research plan.\n"
            "4. Prioritize identifying new angles of investigation rather than rephrasing existing queries.\n"
            "5. Follow the exact categories specified in the research plan and generate queries strictly within those categories.\n"
            "6. If the research plan is broad, generate a diverse set of queries covering different aspects of the investigation.\n"
            "7. If the research plan is highly specific, ensure that the generated queries align with the detailed instructions while maintaining uniqueness.\n\n"

            "### Output Format:\n"
            "Return the generated queries as a JSON object where:\n"
            "- The keys match the exact research categories specified in the research plan.\n"
            "- The values are lists of well-structured search queries.\n\n"

            "### Constraints:\n"
            "- Ensure that each generated query is meaningfully different from previous queries.\n"
            "- Avoid minor rewordings or trivial variations of past queries.\n"
            "- Queries should be optimized to capture new and relevant information about the company.\n\n"

            "Your role is to enhance the depth of the investigation by formulating well-structured, non-redundant, and insightful search queries "
            "strictly based on the provided research plan."
        )

        input_message = (
            f"Company: {company}\n"
            f"Industry: {industry}\n\n"
            f"Research Plan:\n{json.dumps(research_plan, indent=4)}\n\n"
            f"Previously Generated Queries:\n{json.dumps(query_history, indent=4)}\n\n"
            f"Generate new targeted search queries strictly following the research plan. "
            f"Ensure that the queries are meaningfully different from the previously generated ones "
            f"to avoid redundancy while capturing new and relevant information about {company}."
        )


        messages = [
            ("system", system_prompt),
            ("human", input_message)
        ]
        response = llm.invoke(messages)

        print(f"RAW RESPONSE: {response.content}")
        response_content = response.content.strip()
        
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content

        query = json.loads(json_content)
        print(f"[Research Agent] Generated {sum(len(v) for v in query.values())} queries across {len(query)} categories")
        return query

    except Exception as e:
        print(f"[Research Agent] Error in query generation: {e}")
        print(traceback.format_exc())
        return {f"{research_plan['objective']}": f"{research_plan['objective']}"}

def generate_targeted_queries(company: str, industry: str) -> Dict[str, List[str]]:
    """
    Generate targeted search queries with a focus on negative events and legal issues,
    organized by category.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        
        system_prompt = (
            "You are an expert forensic investigator specializing in corporate misconduct research. "
            "Generate targeted search queries to uncover potential issues for a given company, focusing on negative events. "
            "Prioritize cases where the company is the accused party, not the victim.\n\n"
            
            "Key areas of focus:\n"
            "1. Bribery allegations, corruption cases, and improper payments\n"
            "2. Legal cases against the company (not cases filed by the company)\n"
            "3. Fraud or misconduct committed by the company (not fraud against the company)\n"
            "4. Regulatory investigations, fines, and penalties from authorities\n"
            "5. Allegations of unethical business practices or corporate misgovernance\n"
            "6. Financial mismanagement, tax evasion, accounting irregularities, or corporate fraud\n\n"
            
            "Generate search queries in the following categories:\n"
            "- 'bribery_corruption': Bribery cases, corruption scandals, and violations of anti-corruption laws\n"
            "- 'legal_issues': Court cases, litigation, and criminal proceedings against the company\n"
            "- 'regulatory_actions': Investigations, fines, and penalties from regulators\n"
            "- 'management_misconduct': Executive misconduct, unethical business practices, and corporate misgovernance\n"
            "- 'corporate_governance': Board-level disputes, promoter conflicts, and governance failures\n"
            "- 'financial_irregularities': Tax evasion, accounting fraud, financial mismanagement, and loan defaults\n\n"
            
            "For each category, generate 2-3 specific search queries using proper search operators. "
            "Return the output as a JSON object where the keys are category names and the values are lists of query strings. "
            "Ensure the queries are precise and capture events where the company is the accused party."
        )
        
        input_message = (
            f"Company: {company}\n"
            f"Industry: {industry}\n\n"
            f"Generate targeted search queries to discover potential misconduct or issues at {company}."
        )
        
        messages = [
            ("system", system_prompt),
            ("human", input_message)
        ]
        
        response = llm.invoke(messages)
        print(f"RAW RESPONSE: {response.content}")
        response_content = response.content.strip()
        
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content
        
        query_categories = json.loads(json_content)
        
        for category, queries in query_categories.items():
            query_categories[category] = [
                q if company.lower() in q.lower() else f'"{company}" {q}' 
                for q in queries
            ]
        
        print(f"[Research Agent] Generated {sum(len(v) for v in query_categories.values())} queries across {len(query_categories)} categories")
        return query_categories
        
    except Exception as e:
        print(f"[Research Agent] Error in query generation: {e}")
        print(traceback.format_exc())
        
        return {
            "bribery_corruption": [
                f'"{company}" bribery OR corruption OR "illegal payments"'
            ],
            "legal_issues": [
                f'"{company}" lawsuit OR litigation OR "court case against"',
                f'"{company}" "legal dispute" OR "case filed against"',
                f'"{company}" "consumer complaint" OR "business malpractice case"'
            ],
            "regulatory_actions": [
                f'"{company}" "regulatory investigation" OR "penalty imposed" OR "license cancelled"',
                f'"{company}" "non-compliance" OR "violation of regulations" OR "fraudulent practices"',
                f'"{company}" "government action" OR "breach of business laws" OR "policy violation"'
            ],
            "management_misconduct": [
                f'"{company}" "executive scandal" OR "CEO misconduct" OR "ethics breach"',
                f'"{company}" "sexual harassment allegation" OR "workplace misconduct" OR "unfair treatment"',
                f'"{company}" "misuse of company funds" OR "conflict of interest" OR "leadership controversy"'
            ],
            "corporate_governance": [
                f'"{company}" "boardroom conflict" OR "director dispute" OR "corporate governance failure"',
                f'"{company}" "investor concerns" OR "mismanagement" OR "lack of transparency"',
                f'"{company}" "insider dealings" OR "governance irregularities" OR "business ethics issues"'
            ],
            "financial_irregularities": [
                f'"{company}" "accounting fraud" OR "misrepresentation of financials"',
                f'"{company}" "loan default" OR "bad debt" OR "non-performing asset"',
                f'"{company}" "fund diversion" OR "tax evasion" OR "financial mismanagement"'
            ],
            "general": [
                f'"{company}" scandal OR controversy OR "negative news"',
                f'"{company}" "allegations" OR "business criticism" OR "customer complaints"',
                f'"{company}" "labour unrest" OR "public backlash" OR "boycott calls"'
            ]
        }
    
def generate_missing_data_queries(company, data_requirements):
    """
    Generate optimized Google search query to find specific missing company data.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        
        system_prompt = (
            "You are an expert data researcher specializing in finding specific business information. "
            "Your task is to create ONE highly optimized Google search query that will help locate "
            "precise missing data about a company. The query should:\n\n"
            "1. Use quotes for exact phrases and terms that must appear together\n"
            "2. Use OR/AND operators appropriately to expand or narrow results\n"
            "3. Use intext: operator when relevant to ensure specific terms appear in the content\n"
            "4. Focus on specific keywords related to the data required\n"
            "5. Include industry-specific terminology when appropriate\n"
            "6. Include year or date information if time-relevant\n\n"
            
            "DO NOT use site:, filetype:, or intitle: operators in your query.\n\n"
            
            "Based on the data requirements text, determine:\n"
            "- The specific type of company information needed (financial metrics, personnel, strategies, etc.)\n"
            "- The time period relevance (current, historical, specific year)\n"
            "- The level of detail required (high-level overview vs. specific metrics)\n\n"
            
            "Return ONLY the search query string itself, without explanation or additional text."
        )
        
        input_message = (
            f"Company: {company}\n\n"
            f"Data Requirements: {data_requirements}\n\n"
            f"Generate ONE precise Google search query to find this specific information."
        )
        
        messages = [
            ("system", system_prompt),
            ("human", input_message)
        ]
        
        response = llm.invoke(messages)
        query = response.content.strip()
        
        if "```" in query:
            query = query.split("```")[1].strip() if len(query.split("```")) > 1 else query.replace("```", "").strip()
        
        if company.lower() not in query.lower():
            if not query.startswith('"'):
                query = f'"{company}" {query}'
        
        print(f"[Data Retrieval Agent] Generated search query for {company}: {query}")
        return query
        
    except Exception as e:
        print(f"[Data Retrieval Agent] Error in query generation: {e}")
        print(traceback.format_exc())
        
        return None

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
            print(f"[Research Agent] Attempting to parse string response as JSON...")
            try:
                parsed_data = json.loads(raw_output)
                
                if isinstance(parsed_data, list):
                    raw_output = parsed_data
                    print(f"[Research Agent] Successfully parsed string as JSON list with {len(raw_output)} items")
                elif isinstance(parsed_data, dict) and 'organic_results' in parsed_data:
                    raw_output = parsed_data['organic_results']
                    print(f"[Research Agent] Successfully parsed string as JSON object with organic_results")
            except json.JSONDecodeError:
                print(f"[Research Agent] Not valid JSON, treating as text: {raw_output[:100]}...")
                
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
                    
                    print(f"[Research Agent] Created result from text response")
                    return results
        
        if isinstance(raw_output, list):
            for i, item in enumerate(raw_output):
                if isinstance(item, dict) and "title" in item and "link" in item:
                    results.append({
                        "index": i,
                        "title": item["title"].strip(),
                        "link": item["link"].strip(),
                        "date": item.get("date", "Unknown date").strip(),
                        "snippet": item.get("snippet", "").strip(),
                        "source": item.get("source", "Unknown source").strip(),
                        "category": category,
                        "is_quarterly_report": False
                    })
        
    except Exception as e:
        print(f"[Research Agent] Error in parse_serp_results: {e}")
        import traceback
        print(traceback.format_exc())
    
    print(f"[Research Agent] Parsed {len(results)} results from category '{category}'")
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

def group_results(company: str, articles: List[Dict], industry: str = None) -> Dict[str, List[Dict]]:
    """
    Group news articles into event clusters with special handling for quarterly reports
    and improved prioritization of negative events.
    """
    if not articles:
        print("[Research Agent] No articles to cluster")
        return {}
    
    quarterly_report_articles = [a for a in articles if a.get("is_quarterly_report", False)]
    other_articles = [a for a in articles if not a.get("is_quarterly_report", False)]
    
    print(f"[Research Agent] Identified {len(quarterly_report_articles)} quarterly report articles")
    print(f"[Research Agent] Processing {len(other_articles)} non-quarterly report articles")
    
    regular_events = {}
    if other_articles:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            
            system_prompt = (
                "You are an expert in analyzing and clustering news articles related to corporate misconduct, regulatory actions, and financial fraud in India. "
                "Your task is to group articles into distinct event clusters representing specific incidents, regulatory investigations, lawsuits, financial scams, bribery cases, or other forensically relevant matters.\n\n"

                "IMPORTANT GUIDELINES:\n"
                "1. Prioritize negative events where the company is accused of wrongdoing.\n"
                "2. Give special attention to lawsuits and regulatory actions AGAINST the company (not by the company).\n"
                "3. Focus on fraud, bribery, and financial irregularities COMMITTED BY the company (not against it).\n"
                "4. Distinguish between different lawsuits, regulatory actions, and financial penalties.\n"
                "5. Consider the timeframe of events while clustering. If a company has faced multiple penalties, bribery allegations, or lawsuits over different periods, treat them as separate events.\n"
                "6. Ensure that clustered articles refer to the same incident, avoiding broad generalizations.\n"
                "7. Create specific event names with dates and severity levels (High/Medium/Low).\n\n"

                "EVENT NAMING FORMAT:\n"
                "[Event Type]: [Specific Description] ([Date/Timeframe if available]) - [Severity if known]\n\n"

                "Example event names:\n"
                "- 'Fraud Investigation: SEBI Probe into Insider Trading (March 2023) - High'\n"
                "- 'Regulatory Fine: RBI Penalizes Bank for KYC Violations (2021) - Medium'\n"
                "- 'ED Raid: Money Laundering Investigation Against XYZ Group (July 2023) - High'\n"
                "- 'CBI Investigation: Corruption and Bribery Case in Infrastructure Project (2020) - High'\n"
                "- 'Bribery Scandal: Allegations Against XYZ Corp for Government Contract (Q4 2022) - High'\n\n"

                "Return a JSON object where:\n"
                "- Each key is a precise event name following the format above.\n"
                "- Each value is a list of INTEGER indices from the article list.\n"
            )


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

            input_message = (
                f"Company: {company}\n"
                f"Industry: {industry if industry else 'Unknown'}\n"
                f"Articles to analyze: {json.dumps(simplified_articles)}\n\n"
                f"Group these articles about {company} into distinct event clusters, focusing on negative events, lawsuits against the company, and potential misconduct."
            )

            messages = [
                ("system", system_prompt),
                ("human", input_message)
            ]

            response = llm.invoke(messages)
            response_content = response.content.strip()
            
            if "```json" in response_content:
                json_content = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_content = response_content.split("```")[1].strip()
            else:
                json_content = response_content
                
            clustered_indices = json.loads(json_content)
            
            for event_name, indices in clustered_indices.items():
                valid_indices = []
                for idx in indices:
                    if isinstance(idx, str) and idx.isdigit():
                        idx = int(idx)
                    if isinstance(idx, int) and 0 <= idx < len(other_articles):
                        valid_indices.append(idx)
                
                if valid_indices:
                    regular_events[event_name] = [other_articles[i] for i in valid_indices]
            
            print(f"[Research Agent] Grouped non-quarterly articles into {len(regular_events)} events")
            
        except Exception as e:
            print(f"[Research Agent] Error clustering non-quarterly articles: {e}")
            print(traceback.format_exc())
            
            for i, article in enumerate(other_articles):
                event_name = f"News: {article['title'][:50]}..."
                regular_events[event_name] = [article]
    
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
            except:
                date_str = f" ({valid_dates[0]})"
        
        quarterly_event_name = f"Financial Reporting: Quarterly/Annual Results{date_str} - Low"
        regular_events[quarterly_event_name] = quarterly_report_articles
        print(f"[Research Agent] Created a consolidated event for {len(quarterly_report_articles)} quarterly report articles")
    
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
    
    print(f"[Research Agent] Assigned importance scores to {len(importance_scores)} events")
    for event, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"[Research Agent] Event: '{event}' - Score: {score}")
    
    return final_events

def research_agent(state: Dict) -> Dict:
    """
    Enhanced research agent that handles both preliminary and targeted research.
    Prevents duplicate queries and efficiently processes search results.
    """
    print("[Research Agent] Received state:", state)
    
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
        print("[Research Agent] ERROR: Missing company name")
        return {**state, "goto": "meta_agent", "error": "Missing company name"}
        
    if not research_plans:
        print("[Research Agent] ERROR: No research plan provided")
        return {**state, "goto": "meta_agent", "error": "No research plan provided"}
    
    # Get the most recent research plan
    current_plan = research_plans[-1]
    print(f"[Research Agent] Processing research plan: {current_plan.get('objective', 'No objective specified')}")
    
    # Check if this is an event-specific targeted research plan
    target_event = current_plan.get("event_name", None)
    if target_event:
        print(f"[Research Agent] This is a targeted research plan for event: {target_event}")
    
    # Set up output tracking
    all_articles = []
    executed_queries = []
    
    # Generate search queries based on the research plan
    query_categories = generate_queries(company, industry, current_plan, search_history)
    
    # Execute searches and collect results
    for category, queries in query_categories.items():
        print(f"[Research Agent] Processing category: {category}")
        for query in queries:
            # Skip duplicate queries that have been executed before
            if query in [q for sublist in search_history for q in sublist]:
                print(f"[Research Agent] Skipping duplicate query: {query}")
                continue
            
            print(f"[Research Agent] Executing search query: {query}")
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
                "num": "100", 
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
                print(f"[Research Agent] Error executing query '{query}': {e}")
                import traceback
                print(traceback.format_exc())
    
    # Update search history with the executed queries
    search_history.append(executed_queries)
    state["search_history"] = search_history
    
    print(f"[Research Agent] Collected {len(all_articles)} total articles across all categories")
    
    # Handle empty results with a fallback query
    if not all_articles:
        print("[Research Agent] No articles found with targeted queries. Trying fallback query.")
        try:
            fallback_query = f'"{company}" negative news'
            if fallback_query not in [q for sublist in search_history for q in sublist]:
                params = {
                    "engine": "google",
                    "q": fallback_query,
                    "num": "100",
                    "location": "India",
                    "google_domain": "google.co.in",
                    "gl": "in"
                }
                
                search_history[-1].append(fallback_query)
                serp = SerpAPIWrapper(params=params)
                raw_output = serp.run(fallback_query)
                fallback_articles = parse_serp_results(raw_output, "general")
                all_articles.extend(fallback_articles)
                print(f"[Research Agent] Fallback query returned {len(fallback_articles)} articles")
        except Exception as e:
            print(f"[Research Agent] Error with fallback query: {e}")
    
    # Deduplicate articles by URL
    unique_articles = []
    seen_urls = set()
    for article in all_articles:
        if article["link"] not in seen_urls:
            seen_urls.add(article["link"])
            unique_articles.append(article)
    
    print(f"[Research Agent] Deduplicated to {len(unique_articles)} unique articles")
    
    # Process results based on whether this is targeted research for a specific event
    if target_event and return_type == "clustered":
        # For targeted event research, add new articles to the existing event
        if target_event in current_results:
            existing_articles = current_results[target_event]
            existing_urls = {article["link"] for article in existing_articles}
            
            # Only add new articles that aren't already in this event
            new_articles = [a for a in unique_articles if a["link"] not in existing_urls]
            current_results[target_event].extend(new_articles)
            
            print(f"[Research Agent] Added {len(new_articles)} new articles to event: {target_event}")
        else:
            # Create a new event entry if it doesn't exist
            current_results[target_event] = unique_articles
            print(f"[Research Agent] Created new event '{target_event}' with {len(unique_articles)} articles")
        
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
            print(f"[Research Agent] Grouped articles into {len(final_results)} distinct events")
        elif return_type != "clustered":
            # Return unclustered results if requested
            state["research_results"] = unique_articles
            print(f"[Research Agent] Returning {len(unique_articles)} unclustered articles")
    
    return {**state, "goto": "meta_agent"}
