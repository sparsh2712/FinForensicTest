import json
import os
import time
import logging
import yaml
from typing import Dict, List, Any, Optional
import requests

logger = logging.getLogger("corporate_agent")

class NSEToolSimple:
    """Simplified version of NSE Tool for corporate data fetching"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.base_url = self.config.get("base_url", "https://www.nseindia.com")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.company = config.get("company", "")
        self.symbol = config.get("symbol", "")
    
    def fetch_data(self, stream_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch data from NSE for a specific stream type"""
        params = params or {}
        
        # Add company/symbol to params if not present
        if "symbol" in params and not params["symbol"] and self.symbol:
            params["symbol"] = self.symbol
            
        if "company" in params and not params["company"] and self.company:
            params["company"] = self.company
            
        # Mock API endpoints based on stream type
        endpoints = {
            "BoardMeetings": "/api/corporates/boardmeeting",
            "Announcements": "/api/corporates/announcements",
            "CorporateActions": "/api/corporates/corporate-actions",
            "AnnualReports": "/api/corporates/annual-reports",
            "FinancialResults": "/api/corporates/financial-results"
        }
        
        endpoint = endpoints.get(stream_type)
        if not endpoint:
            logger.warning(f"Unknown stream type: {stream_type}")
            return []
            
        # For demo purposes, we'll return mock data instead of making actual API calls
        mock_data = self._get_mock_data(stream_type)
        
        # Simulate API delay
        time.sleep(0.5)
        
        return mock_data
    
    def _get_mock_data(self, stream_type: str) -> List[Dict[str, Any]]:
        """Get mock data for a stream type"""
        company = self.company
        
        if stream_type == "BoardMeetings":
            return [
                {"meetingDate": "2024-02-15", "purpose": "Financial Results", "company": company},
                {"meetingDate": "2024-01-05", "purpose": "Fund Raising", "company": company}
            ]
        elif stream_type == "Announcements":
            return [
                {"announcementDate": "2024-03-10", "subject": "Quarterly Results", "company": company},
                {"announcementDate": "2024-02-20", "subject": "New Product Launch", "company": company}
            ]
        elif stream_type == "CorporateActions":
            return [
                {"exDate": "2024-02-28", "purpose": "Dividend", "company": company},
                {"exDate": "2023-11-15", "purpose": "Bonus", "company": company}
            ]
        elif stream_type == "AnnualReports":
            return [
                {"year": "2023", "reportLink": "#", "company": company},
                {"year": "2022", "reportLink": "#", "company": company}
            ]
        elif stream_type == "FinancialResults":
            return [
                {"period": "Q4 2023", "revenue": "1000 Cr", "profit": "100 Cr", "company": company},
                {"period": "Q3 2023", "revenue": "950 Cr", "profit": "90 Cr", "company": company}
            ]
        else:
            return []

def load_governance_data(company: str, symbol: str = None) -> Dict[str, Any]:
    """Load corporate governance data"""
    # In a real implementation, this would load from a database or API
    # For demo purposes, we'll return mock data
    
    return {
        "board_composition": {
            "independent_directors": 5,
            "executive_directors": 3,
            "women_directors": 2,
            "board_size": 8
        },
        "committees": {
            "audit_committee": {"size": 3, "independent": 3},
            "nomination_committee": {"size": 3, "independent": 2},
            "risk_committee": {"size": 3, "independent": 2}
        },
        "compliance": {
            "sebi_compliance": "Compliant",
            "listing_compliance": "Compliant",
            "fema_compliance": "Compliant"
        },
        "key_metrics": {
            "promoter_holding": "45%",
            "institutional_holding": "35%",
            "public_holding": "20%"
        }
    }

def get_company_symbol(company_name: str) -> str:
    """Get company symbol from name (simplified)"""
    # In a real implementation, this would query a database or API
    # For demonstration, we'll use a simplified approach
    
    # Sample symbol mapping
    symbol_map = {
        "Reliance Industries": "RELIANCE",
        "Tata Consultancy Services": "TCS",
        "HDFC Bank": "HDFCBANK",
        "Infosys": "INFY",
        "ITC": "ITC",
        "Bharti Airtel": "BHARTIARTL",
        "Hindustan Unilever": "HINDUNILVR"
    }
    
    # Try exact match
    if company_name in symbol_map:
        return symbol_map[company_name]
    
    # Try partial match
    for name, symbol in symbol_map.items():
        if name.lower() in company_name.lower() or company_name.lower() in name.lower():
            return symbol
    
    # Return a capitalized version of the company name as fallback
    return company_name.upper().replace(" ", "")[:10]

def get_default_stream_config() -> Dict[str, Dict[str, Any]]:
    """Get default stream configuration"""
    return {
        "BoardMeetings": {
            "active": True,
            "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
        },
        "Announcements": {
            "active": True,
            "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
        },
        "CorporateActions": {
            "active": True,
            "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
        },
        "AnnualReports": {
            "active": True,
            "input_params": {}
        },
        "FinancialResults": {
            "active": True,
            "input_params": {"from_date": "01-01-2023", "to_date": "31-12-2023"}
        }
    }

def corporate_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified corporate agent that collects corporate governance data.
    
    Args:
        state: The current state dictionary containing:
            - company: Company name
            - corporate_streams: List of stream types to collect
            - corporate_symbol: Optional company symbol
    
    Returns:
        Updated state containing corporate data and next routing information
    """
    logger.info(f"Starting corporate agent for {state.get('company')}")
    
    try:
        company = state.get("company", "")
        if not company:
            logger.error("Company name is missing!")
            return {**state, "goto": "meta_agent", "corporate_status": "ERROR", "error": "Company name is missing"}
        
        # Get the company symbol if not provided
        symbol = state.get("corporate_symbol")
        if not symbol:
            symbol = get_company_symbol(company)
            state["corporate_symbol"] = symbol
        
        logger.info(f"Using symbol {symbol} for company {company}")
        
        # Create an NSE tool instance with company info
        nse_config = {"company": company, "symbol": symbol}
        nse_tool = NSEToolSimple(nse_config)
        
        # Get the stream configuration
        stream_config = state.get("corporate_stream_config", get_default_stream_config())
        
        # Collect data from each active stream
        corporate_data = {}
        for stream_name, stream_info in stream_config.items():
            if stream_info.get("active", False):
                logger.info(f"Collecting data for stream: {stream_name}")
                try:
                    stream_data = nse_tool.fetch_data(
                        stream_name, 
                        stream_info.get("input_params", {})
                    )
                    corporate_data[stream_name] = stream_data
                    logger.info(f"Collected {len(stream_data)} items from {stream_name}")
                except Exception as e:
                    logger.error(f"Error collecting data for stream {stream_name}: {str(e)}")
                    corporate_data[stream_name] = []
        
        # Load corporate governance data
        governance_data = load_governance_data(company, symbol)
        
        # Prepare the results
        corporate_results = {
            "success": True,
            "company": company,
            "symbol": symbol,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "governance": governance_data,
            "data": corporate_data,
            "summary": {
                "total_streams": len(stream_config),
                "stream_counts": {stream: len(data) for stream, data in corporate_data.items()}
            }
        }
        
        # Update state with results
        state["corporate_results"] = corporate_results
        state["corporate_status"] = "DONE"
        
        # If synchronous_pipeline is set, use the next_agent value, otherwise go to meta_agent
        goto = "meta_agent"
        if state.get("synchronous_pipeline", False):
            goto = state.get("next_agent", "meta_agent")
        
        logger.info(f"Corporate agent completed successfully for {company}")
        return {**state, "goto": goto}
    
    except Exception as e:
        logger.error(f"Error in corporate agent: {str(e)}")
        return {
            **state,
            "goto": "meta_agent",
            "corporate_status": "ERROR",
            "error": f"Error in corporate agent: {str(e)}"
        }