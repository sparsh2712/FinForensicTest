import requests
import json
import brotli
import yaml
import os
from typing import Dict
from urllib.parse import quote
from tenacity import retry, stop_after_attempt, wait_exponential

class NSETool:
    def __init__(self, config):
        self.config = config
        self.config.setdefault("base_url", "https://www.nseindia.com")
        self.config.setdefault("refresh_interval", 25)
        self.config.setdefault("config_path", "assets/nse_config.yaml")
        self.config.setdefault("headers_path", "assets/headers.yaml")
        self.config.setdefault("cookie_path", "assets/cookies.yaml")
        self.config.setdefault("schema_path", "assets/nse_schema.yaml")
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
        self._create_new_session()
    
    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    
    def _create_new_session(self):
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        if self.config["use_hardcoded_cookies"]:
            for name, value in self.cookies.items():
                self.session.cookies.set(name, value, domain=self.config["domain"])
        else:
            self.session.get(self.config["base_url"])
            filings_url = f"{self.config['base_url']}/companies-listing/corporate-filings-announcements"
            self.session.get(filings_url)
            self.cookies = self.session.cookies.get_dict()
            
        return self.session

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10))
    def _refresh_session(self, referer):
        if self.session is None:
            self._create_new_session()
        
        headers = self.headers.copy()
        headers["Referer"] = referer
        
        self.session.get(referer, headers=headers, timeout=10)
            
        if not self.config["use_hardcoded_cookies"]:
            self.cookies = self.session.cookies.get_dict()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def make_request(self, url, referer):
        self._refresh_session(referer)
        
        response = self.session.get(url, headers=self.headers, timeout=60)
        response.raise_for_status()
        
        content = response.content
        if not content:
            return None

        if 'br' in response.headers.get('Content-Encoding', ''):
            decompressed_content = brotli.decompress(content)
            json_text = decompressed_content.decode('utf-8')
            return json.loads(json_text)
        else:
            return response.json()

    def fetch_data_from_nse(self, stream, input_params):
        stream_config = self.data_config.get(stream)
        if not stream_config:
            return []
        
        params = stream_config.get('params', {}).copy() 
        if "issuer" in params:
            params["issuer"] = self.config["company"]
        if "symbol" in params:
            params["symbol"] = self.config["symbol"]
        params.update(input_params)

        url = self._construct_url(stream_config.get("endpoint"), params)
        result = self.make_request(url, stream_config.get("referer", self.fallback_referer))
        
        if result:
            max_results = input_params.get("max_results", 20)
            
            if isinstance(result, list):
                data_list = result
                if max_results and len(data_list) > max_results:
                    data_list = data_list[:max_results]
                return data_list
            elif isinstance(result, dict):
                data_list = result.get("data", result)
                if isinstance(data_list, list):
                    if max_results and len(data_list) > max_results:
                        data_list = data_list[:max_results]
                    return data_list
                else:
                    return [result]
            return [result]
        else:
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
        if self.session:
            self.session.close()

    def _filter_on_schema(self, data, schema):
        if not data or not isinstance(data, list):
            return []
        return [{new_key: entry[old_key] for new_key, old_key in schema.items() if old_key in entry} for entry in data]
    
    def _get_schema(self, stream_name):
        schema_data = self._load_yaml(self.config["schema_path"])
        return schema_data.get(stream_name, {})
    
    def _process_stream(self, stream, params, schema):
        data = self.fetch_data_from_nse(stream, params)
        filtered_data = self._filter_on_schema(data, schema)
        return filtered_data
    
    def _process_announcements(self, params, schema):
        return self._process_stream("Announcements", params, schema)
    
    def _process_ann_xbrl(self, params, schema):
        data = self.fetch_data_from_nse("AnnXBRL", params)
        filtered_data = self._filter_on_schema(data, schema)
        for entry in filtered_data:
            app_id = entry.get("appId")
            if app_id:
                entry["details"] = self._get_announcement_details(app_id, params)
        return filtered_data
    
    def _get_announcement_details(self, appId, params):
        try:
            data = self.fetch_data_from_nse("AnnXBRLDetails", {"appId": appId, "type": params.get("type", "announcements")})
            return data[0] if data else {}
        except:
            return {}
    
    def _process_annual_reports(self, params, schema):
        return self._process_stream("AnnualReports", params, schema)
    
    def _process_esg_reports(self, params, schema):
        return self._process_stream("BussinessSustainabilitiyReport", params, schema)
    
    def _process_board_meetings(self, params, schema):
        return self._process_stream("BoardMeetings", params, schema)
    
    def _process_corporate_actions(self, params, schema):
        return self._process_stream("CorporateActions", params, schema)
    
    def get(self, streams=None, stream_params=None):
        if stream_params is None:
            stream_params = {}
            
        if streams is None:
            streams = list(self.stream_processors.keys())
            
        if isinstance(streams, str):
            streams = [streams]
            
        result = {}
        
        for stream in streams:
            if stream in self.stream_processors:
                try:
                    schema = self._get_schema(stream)
                    processor = self.stream_processors[stream]
                    params = stream_params.get(stream, {})
                    result[stream] = processor(params, schema)
                except Exception:
                    result[stream] = []
                    
        return result

def corporate_agent(state: Dict) -> Dict:
    print("[Corporate Agent] Starting corporate data collection process...")
    
    company = state.get("company")
    symbol = state.get("company_symbol", company)
    
    if not company:
        print("[Corporate Agent] ERROR: Company name is missing!")
        return {**state, "goto": "meta_agent", "corporate_status": "ERROR", "error": "Company name is required"}
    
    params_path = "assets/params.yaml"
    
    try:
        with open(params_path, "r") as f:
            yaml_params = yaml.safe_load(f)
    except Exception as e:
        print(f"[Corporate Agent] ERROR: Failed to load params file: {e}")
        return {**state, "goto": "meta_agent", "corporate_status": "ERROR", "error": f"Failed to load params file: {e}"}
    
    config = {
        "company": company,
        "symbol": symbol
    }
    
    try:
        nse_tool = NSETool(config)
        
        streams = ["Announcements", "AnnXBRL", "AnnualReports", "BussinessSustainabilitiyReport", "BoardMeetings", "CorporateActions"]
        
        corporate_results = nse_tool.get(streams, yaml_params)
        nse_tool.close()
        
        state["corporate_results"] = corporate_results
        state["corporate_status"] = "DONE"
        
        print(f"[Corporate Agent] Corporate data collection complete for {company}.")
        
    except Exception as e:
        print(f"[Corporate Agent] ERROR during corporate data collection: {e}")
        import traceback
        print(traceback.format_exc())
        return {**state, "goto": "meta_agent", "corporate_status": "ERROR", "error": f"Error during corporate data collection: {str(e)}"}
    
    return {**state, "goto": "meta_agent"}