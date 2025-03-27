import requests
import json
import brotli
import yaml
import os
import logging
from typing import Dict
from urllib.parse import quote
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("corporate_agent")

class NSETool:
    def __init__(self, config):
        self.config = config
        self.config.setdefault("base_url", "https://www.nseindia.com")
        self.config.setdefault("refresh_interval", 25)
        # Use relative paths with os.path to ensure portability
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config.setdefault("config_path", os.path.join(base_dir, "assets", "nse_config.yaml"))
        self.config.setdefault("headers_path", os.path.join(base_dir, "assets", "headers.yaml"))
        self.config.setdefault("cookie_path", os.path.join(base_dir, "assets", "cookies.yaml"))
        self.config.setdefault("schema_path", os.path.join(base_dir, "assets", "nse_schema.yaml"))
        self.config.setdefault("use_hardcoded_cookies", False)
        self.config.setdefault("domain", "nseindia.com")
        
        self.data_config = self._load_yaml(self.config["config_path"])
        self.headers = self._load_yaml(self.config["headers_path"])
        self.cookies = self._load_yaml(self.config["cookie_path"]) if self.config["use_hardcoded_cookies"] else None
        self.session = None
        self.fallback_referer = "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings"
        self.stream_processors = {
            "Announcements": self._process_announcements,
            "AnnXBRL": self._process_ann_xbrl,
            "AnnualReports": self._process_annual_reports,
            "BussinessSustainabilitiyReport": self._process_esg_reports,
            "BoardMeetings": self._process_board_meetings,
            "CorporateActions": self._process_corporate_actions,
        }
        logger.info(f"Initializing NSETool for company: {self.config.get('company')}")
        self._create_new_session()
    
    def _load_yaml(self, path):
        logger.debug(f"Loading YAML from: {path}")
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading YAML from {path}: {e}")
            raise
    
    def _create_new_session(self):
        logger.info("Creating new HTTP session")
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        if self.config["use_hardcoded_cookies"]:
            logger.debug("Using hardcoded cookies")
            for name, value in self.cookies.items():
                self.session.cookies.set(name, value, domain=self.config["domain"])
        else:
            logger.debug("Fetching fresh cookies from NSE")
            self.session.get(self.config["base_url"])
            filings_url = f"{self.config['base_url']}/companies-listing/corporate-filings-announcements"
            self.session.get(filings_url)
            self.cookies = self.session.cookies.get_dict()
            logger.debug(f"Retrieved {len(self.cookies)} cookies")
            
        return self.session

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10))
    def _refresh_session(self, referer):
        if self.session is None:
            logger.info("Session is None, creating new session")
            self._create_new_session()
        
        headers = self.headers.copy()
        headers["Referer"] = referer
        
        logger.debug(f"Refreshing session with referer: {referer}")
        try:
            self.session.get(referer, headers=headers, timeout=10)
            logger.debug("Session refreshed successfully")
        except Exception as e:
            logger.warning(f"Error refreshing session: {e}")
            raise
            
        if not self.config["use_hardcoded_cookies"]:
            self.cookies = self.session.cookies.get_dict()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def make_request(self, url, referer):
        self._refresh_session(referer)
        
        logger.debug(f"Making request to: {url}")
        try:
            response = self.session.get(url, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            content = response.content
            if not content:
                logger.warning("Received empty response")
                return None

            if 'br' in response.headers.get('Content-Encoding', ''):
                logger.debug("Decompressing brotli content")
                decompressed_content = brotli.decompress(content)
                json_text = decompressed_content.decode('utf-8')
                return json.loads(json_text)
            else:
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in make_request: {e}")
            raise

    def fetch_data_from_nse(self, stream, input_params):
        stream_config = self.data_config.get(stream)
        if not stream_config:
            logger.warning(f"Stream configuration not found for: {stream}")
            return []
        
        params = stream_config.get('params', {}).copy() 
        if "issuer" in params:
            params["issuer"] = self.config["company"]
        if "symbol" in params:
            params["symbol"] = self.config["symbol"]
        params.update(input_params)

        url = self._construct_url(stream_config.get("endpoint"), params)
        logger.info(f"Fetching {stream} data for {self.config.get('company')}")
        logger.debug(f"Request URL: {url}")
        
        try:
            result = self.make_request(url, stream_config.get("referer", self.fallback_referer))
            
            # Log NSE response data
            if result:
                # Format the first part of response for logging (limit size)
                response_preview = json.dumps(result)[:1000]
                if len(json.dumps(result)) > 1000:
                    response_preview += "..."
                logger.info(f"NSE {stream} Response Preview: {response_preview}")
                
                # Log data type and structure
                if isinstance(result, list):
                    logger.info(f"NSE {stream} Response: List with {len(result)} items")
                    if result and len(result) > 0:
                        sample_keys = list(result[0].keys()) if isinstance(result[0], dict) else "non-dict items"
                        logger.info(f"NSE {stream} Sample Keys: {sample_keys}")
                elif isinstance(result, dict):
                    logger.info(f"NSE {stream} Response: Dict with keys {list(result.keys())}")
                    if "data" in result and isinstance(result["data"], list):
                        logger.info(f"NSE {stream} Data: List with {len(result['data'])} items")
                        if result["data"] and len(result["data"]) > 0:
                            sample_keys = list(result["data"][0].keys()) if isinstance(result["data"][0], dict) else "non-dict items"
                            logger.info(f"NSE {stream} Sample Data Keys: {sample_keys}")
                
                max_results = input_params.get("max_results", 20)
                
                if isinstance(result, list):
                    data_list = result
                    if max_results and len(data_list) > max_results:
                        data_list = data_list[:max_results]
                    logger.info(f"Retrieved {len(data_list)} items for {stream}")
                    return data_list
                elif isinstance(result, dict):
                    data_list = result.get("data", result)
                    if isinstance(data_list, list):
                        if max_results and len(data_list) > max_results:
                            data_list = data_list[:max_results]
                        logger.info(f"Retrieved {len(data_list)} items for {stream}")
                        return data_list
                    else:
                        logger.info(f"Retrieved data object for {stream}")
                        return [result]
                return [result]
            else:
                logger.warning(f"No data returned for {stream}")
                return []
        except Exception as e:
            logger.error(f"Error fetching data for {stream}: {e}")
            return []
            
    def _construct_url(self, endpoint, params):
        filtered_params = {k: v for k, v in params.items() if v is not None and v != ""}
        
        base_url = f"{self.config['base_url']}/api/{endpoint}"
        
        if filtered_params:
            query_parts = []
            for key, value in filtered_params.items():
                encoded_value = quote(str(value))
                query_parts.append(f"{key}={encoded_value}")
            
            query_string = "&".join(query_parts)
            return f"{base_url}?{query_string}"
        
        return base_url
    
    def close(self):
        logger.info("Closing NSE tool session")
        if self.session:
            self.session.close()

    def _filter_on_schema(self, data, schema):
        if not data or not isinstance(data, list):
            return []
        filtered_data = [{new_key: entry[old_key] for new_key, old_key in schema.items() if old_key in entry} for entry in data]
        logger.debug(f"Filtered data according to schema, resulting in {len(filtered_data)} entries")
        return filtered_data
    
    def _get_schema(self, stream_name):
        logger.debug(f"Getting schema for stream: {stream_name}")
        schema_data = self._load_yaml(self.config["schema_path"])
        return schema_data.get(stream_name, {})
    
    def _process_stream(self, stream, params, schema):
        logger.debug(f"Processing stream: {stream} with params: {params}")
        data = self.fetch_data_from_nse(stream, params)
        filtered_data = self._filter_on_schema(data, schema)
        return filtered_data
    
    def _process_announcements(self, params, schema):
        logger.info("Processing Announcements stream")
        return self._process_stream("Announcements", params, schema)
    
    def _process_ann_xbrl(self, params, schema):
        logger.info("Processing AnnXBRL stream")
        data = self.fetch_data_from_nse("AnnXBRL", params)
        filtered_data = self._filter_on_schema(data, schema)
        for entry in filtered_data:
            app_id = entry.get("appId")
            if app_id:
                logger.debug(f"Fetching XBRL details for appId: {app_id}")
                entry["details"] = self._get_announcement_details(app_id, params)
        return filtered_data
    
    def _get_announcement_details(self, appId, params):
        try:
            logger.debug(f"Getting announcement details for appId: {appId}")
            data = self.fetch_data_from_nse("AnnXBRLDetails", {"appId": appId, "type": params.get("type", "announcements")})
            return data[0] if data else {}
        except Exception as e:
            logger.error(f"Error getting announcement details for appId {appId}: {e}")
            return {}
    
    def _process_annual_reports(self, params, schema):
        logger.info("Processing AnnualReports stream")
        return self._process_stream("AnnualReports", params, schema)
    
    def _process_esg_reports(self, params, schema):
        logger.info("Processing ESG Reports stream")
        return self._process_stream("BussinessSustainabilitiyReport", params, schema)
    
    def _process_board_meetings(self, params, schema):
        logger.info("Processing BoardMeetings stream")
        return self._process_stream("BoardMeetings", params, schema)
    
    def _process_corporate_actions(self, params, schema):
        logger.info("Processing CorporateActions stream")
        return self._process_stream("CorporateActions", params, schema)
    
    def get(self, streams=None, stream_params=None):
        """
        Retrieve data from specified streams with improved error handling for Key Personnel data.
        """
        if stream_params is None:
            stream_params = {}
                
        if streams is None:
            streams = list(self.stream_processors.keys())
                
        if isinstance(streams, str):
            streams = [streams]
                
        logger.info(f"Retrieving data for streams: {streams}")
        result = {}
            
        for stream in streams:
            if stream in self.stream_processors:
                try:
                    schema = self._get_schema(stream)
                    processor = self.stream_processors[stream]
                    params = stream_params.get(stream, {})
                    logger.info(f"Processing stream: {stream}")
                    result[stream] = processor(params, schema)
                    logger.info(f"Retrieved {len(result[stream])} records for {stream}")
                except Exception as e:
                    logger.error(f"Error processing stream {stream}: {e}")
                    result[stream] = []
        
        # Properly load the JSON file with error handling
        try:
            logger.info("Attempting to load Key Personnel data")
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(base_dir, "assets", "corporate_governance.json")
            
            # Check if file exists
            if not os.path.exists(json_path):
                logger.error(f"Key Personnel file not found at: {json_path}")
                result["Key_Personnel"] = {"error": "File not found", "board_of_directors": [], "communities": {}}
                return result
                
            # Open and parse the JSON file
            with open(json_path, 'r') as f:
                key_personnel_dict = json.load(f)
            
            # Get company symbol and validate
            company_symbol = self.config.get("symbol")
            if not company_symbol:
                logger.error("Company symbol is missing")
                result["Key_Personnel"] = {"error": "Company symbol missing", "board_of_directors": [], "communities": {}}
                return result
                
            # Try multiple formats of company symbol to improve matching
            personnel_data = None
            symbol_variants = [
                company_symbol,
                company_symbol.upper(),
                company_symbol.lower(),
                company_symbol.replace(" ", ""),
                company_symbol.strip()
            ]
            
            for variant in symbol_variants:
                if variant in key_personnel_dict:
                    personnel_data = key_personnel_dict[variant]
                    logger.info(f"Found Key Personnel data using symbol variant: {variant}")
                    break
            
            # Handle missing data case with better structure
            if personnel_data is None:
                logger.warning(f"No Key Personnel data found for symbol: {company_symbol} or variants")
                result["Key_Personnel"] = {"board_of_directors": [], "communities": {}}
            else:
                # Ensure data has expected structure
                if isinstance(personnel_data, dict):
                    if "board_of_directors" not in personnel_data:
                        personnel_data["board_of_directors"] = []
                    if "communities" not in personnel_data:
                        personnel_data["communities"] = {}
                    result["Key_Personnel"] = personnel_data
                else:
                    logger.error(f"Unexpected Key Personnel data format for {company_symbol}")
                    result["Key_Personnel"] = {"error": "Unexpected data format", "board_of_directors": [], "communities": {}}
                
            logger.info(f"Successfully added Key Personnel data for {company_symbol}")
            
        except Exception as e:
            logger.error(f"Error loading Key Personnel data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Provide an empty structure to avoid NoneType errors
            result["Key_Personnel"] = {"board_of_directors": [], "communities": {}}
        
        return result

def corporate_agent(state: Dict) -> Dict:
    logger.info("Starting corporate data collection process")
    
    company = state.get("company")
    symbol = state.get("company_symbol", company)
    
    if not company:
        logger.error("Company name is missing")
        return {**state, "goto": "corporate_meta_writer_agent", "corporate_status": "ERROR", "error": "Company name is required"}
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params_path = os.path.join(base_dir, "assets", "params.yaml")
    
    try:
        logger.debug(f"Loading parameters from {params_path}")
        with open(params_path, "r") as f:
            yaml_params = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load params file: {e}")
        return {**state, "goto": "corporate_meta_writer_agent", "corporate_status": "ERROR", "error": f"Failed to load params file: {e}"}
    
    config = {
        "company": company,
        "symbol": symbol
    }
    
    logger.info(f"Initializing NSETool for {company} ({symbol})")
    
    try:
        nse_tool = NSETool(config)
        
        streams = ["Announcements", "AnnXBRL", "AnnualReports", "BussinessSustainabilitiyReport", "BoardMeetings", "CorporateActions"]
        logger.info(f"Requesting data for streams: {streams}")
        
        corporate_results = nse_tool.get(streams, yaml_params)
        nse_tool.close()
        
        # Log summary of results
        for stream, data in corporate_results.items():
            logger.info(f"{stream}: Retrieved {len(data)} records")
        
        state["corporate_results"] = corporate_results
        state["corporate_status"] = "DONE"
        
        logger.info(f"Corporate data collection complete for {company}")
        
    except Exception as e:
        logger.error(f"Error during corporate data collection: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {**state, "goto": "corporate_meta_writer_agent", "corporate_status": "ERROR", "error": f"Error during corporate data collection: {str(e)}"}
    
    return {**state, "goto": "corporate_meta_writer_agent"}