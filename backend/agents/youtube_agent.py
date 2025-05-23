import json
import time
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
import googleapiclient.errors
from tenacity import retry, stop_after_attempt, wait_exponential
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

logger = logging.getLogger("youtube_agent")

RETRY_LIMIT = 3
MULTIPLIER = 2
MIN_WAIT = 0.5
MAX_WAIT = 10

# Load LLM config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)  # Go up one level from /agents to /backend
LLM_CONFIG_PATH = os.path.join(BASE_DIR, "assets", "llm_config.yaml")

try:
    with open(LLM_CONFIG_PATH, 'r') as f:
        LLM_CONFIG = yaml.safe_load(f)
    # Get agent-specific config or fall back to default
    AGENT_CONFIG = LLM_CONFIG.get("youtube_agent", LLM_CONFIG.get("default", {}))
except Exception as e:
    print(f"Error loading LLM config: {e}, using defaults")
    AGENT_CONFIG = {"model": "gemini-2.0-flash", "temperature": 0.0}

# Import LLM and prompt manager for summarization
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.utils.prompt_manager import PromptManager

# Initialize prompt manager
prompt_manager = PromptManager(os.path.join(BASE_DIR, "prompts"))

class YoutubeTool:    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("youtube_agent")
        self.youtube_api_key = config.get("youtube_api_key")
        self.retry_limit = config.get("retry_limit", RETRY_LIMIT)
        self.multiplier = config.get("multiplier", MULTIPLIER)
        self.min_wait = config.get("min_wait", MIN_WAIT)
        self.max_wait = config.get("max_wait", MAX_WAIT)
        
        self.youtube = build(
            "youtube", "v3", developerKey=self.youtube_api_key
        )
        self.transcriptor = YouTubeTranscriptApi()
        self._update_retry_params()
    
    def _update_retry_params(self):
        global RETRY_LIMIT, MULTIPLIER, MIN_WAIT, MAX_WAIT
        RETRY_LIMIT = self.retry_limit
        MULTIPLIER = self.multiplier
        MIN_WAIT = self.min_wait
        MAX_WAIT = self.max_wait
        
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    def get_transcript(self, video_id: str) -> Optional[str]:
        try:
            transcript = self.transcriptor.fetch(video_id)
            self.logger.info(f"Successfully fetched transcript for video: {video_id}")
            return " ".join(snippet.text for snippet in transcript)
        except NoTranscriptFound:
            try:
                transcript_list = self.transcriptor.list_transcripts(video_id)
                fetched_transcript = next(iter(transcript_list)).fetch()
                # Add return statement here
                return " ".join(snippet.text for snippet in fetched_transcript)
            except (NoTranscriptFound, TranscriptsDisabled) as e:
                self.logger.error(f"No transcript found for video {video_id}: {e}")
                return f"Failed to transcribe the video: No transcript available"
        except Exception as e:
            self.logger.error(f"Error fetching transcript: {e}")
            return f"Failed to transcribe the video: {str(e)}"
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    def get_playlists_from_channel(self, channel_id: str, max_results: int = 50) -> List[Dict]:
        playlists = []
        next_page_token = None
        
        try:
            while True:
                response = self.youtube.playlists().list(
                    part="snippet,contentDetails",
                    channelId=channel_id,
                    maxResults=min(50, max_results - len(playlists)),
                    pageToken=next_page_token
                ).execute()
                
                for item in response.get("items", []):
                    playlist_data = {
                        "id": item["id"],
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "video_count": item["contentDetails"]["itemCount"]
                    }
                    playlists.append(playlist_data)
                    
                    if len(playlists) >= max_results:
                        self.logger.info(f"Retrieved {len(playlists)} playlists from channel: {channel_id}")
                        return playlists
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            
            self.logger.info(f"Retrieved {len(playlists)} playlists from channel: {channel_id}")
            return playlists
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error getting playlists: {e}")
            return []
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    def get_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        try:
            response = self.youtube.search().list(
                part="snippet",
                q=channel_name,
                type="channel",
                maxResults=1
            ).execute()
            
            if response.get("items"):
                channel_id = response["items"][0]["id"]["channelId"]
                self.logger.info(f"Found channel ID {channel_id} for channel name: {channel_name}")
                return channel_id
                
            self.logger.warning(f"No channel found for name: {channel_name}")
            return None
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error getting channel ID: {e}")
            return None
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    def get_videos_from_playlist(self, playlist_id: str, max_results: int = 50) -> List[Dict]:
        videos = []
        next_page_token = None
        
        try:
            while True:
                response = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()
                
                for item in response.get("items", []):
                    video_data = {
                        "id": item["contentDetails"]["videoId"],
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "published_at": item["snippet"]["publishedAt"],
                        "channel_title": item["snippet"].get("channelTitle")
                    }
                    videos.append(video_data)
                    
                    if len(videos) >= max_results:
                        self.logger.info(f"Retrieved {len(videos)} videos from playlist: {playlist_id}")
                        return videos
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            
            self.logger.info(f"Retrieved {len(videos)} videos from playlist: {playlist_id}")
            return videos
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error getting videos from playlist: {e}")
            return []
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    def get_video_details(self, video_id: str) -> Optional[Dict]:
        try:
            response = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            ).execute()
            
            if response.get("items"):
                video = response["items"][0]
                details = {
                    "title": video["snippet"]["title"],
                    "description": video["snippet"]["description"],
                    "published_at": video["snippet"]["publishedAt"],
                    "channel_id": video["snippet"]["channelId"],
                    "channel_title": video["snippet"]["channelTitle"],
                    "duration": video["contentDetails"]["duration"],
                    "view_count": int(video["statistics"].get("viewCount", 0)),
                    "like_count": int(video["statistics"].get("likeCount", 0)),
                    "comment_count": int(video["statistics"].get("commentCount", 0))
                }
                
                self.logger.info(f"Retrieved details for video: {video_id}")
                return details
                
            self.logger.warning(f"No details found for video: {video_id}")
            return None
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error getting video details: {e}")
            return None
    
    @retry(stop=stop_after_attempt(RETRY_LIMIT), wait=wait_exponential(multiplier=MULTIPLIER, min=MIN_WAIT, max=MAX_WAIT))
    def search_videos(self, query: str, max_results: int = 50, order: str = "relevance") -> List[Dict]:
        videos = []
        next_page_token = None
        
        try:
            while True:
                response = self.youtube.search().list(
                    part="snippet",
                    q=query,
                    type="video",
                    order=order,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()
                
                for item in response.get("items", []):
                    video_data = {
                        "id": item["id"]["videoId"],
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "channel_title": item["snippet"]["channelTitle"],
                        "published_at": item["snippet"]["publishedAt"]
                    }
                    videos.append(video_data)
                    
                    if len(videos) >= max_results:
                        self.logger.info(f"Found {len(videos)} videos for query: {query}")
                        return videos
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            
            self.logger.info(f"Found {len(videos)} videos for query: {query}")
            return videos
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"API error searching videos: {e}")
            return []

def summarize_transcript(company: str, transcript_data: Dict) -> Dict:
    """
    Summarize a transcript with focus on corporate governance aspects.
    
    Args:
        company: The company name
        transcript_data: Dictionary containing transcript information
    
    Returns:
        Dictionary with original transcript and added summary
    """
    print(f"[YouTube Agent] Summarizing transcript for: {transcript_data.get('title', 'Unknown video')}")
    
    if not transcript_data.get("transcript"):
        print(f"[YouTube Agent] No transcript content to summarize")
        return {**transcript_data, "transcript_summary": "No transcript content available to summarize."}
    
    try:
        # Initialize LLM using the agent config
        llm = ChatGoogleGenerativeAI(
            model=AGENT_CONFIG["model"],
            temperature=AGENT_CONFIG["temperature"]
        )
        
        # Prepare variables for prompt template
        variables = {
            "company": company,
            "video_title": transcript_data.get("title", "Unknown"),
            "transcript": transcript_data.get("transcript", "")[:50000]  # Limit transcript size if needed
        }
        
        # Get prompts from prompt manager
        system_prompt, human_prompt = prompt_manager.get_prompt(
            "youtube_agent", 
            "transcript_summary", 
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
        
        print(f"[YouTube Agent] Successfully summarized transcript, summary length: {len(summary)} chars")
        
        # Return data with added summary
        return {
            **transcript_data, 
            "transcript_summary": summary
        }
        
    except Exception as e:
        print(f"[YouTube Agent] Error summarizing transcript: {e}")
        # Return original data with error message
        return {
            **transcript_data, 
            "transcript_summary": f"Error during summarization: {str(e)}"
        }

def youtube_agent(state: Dict) -> Dict:
    print("[YouTube Agent] Starting YouTube processing...")
    
    youtube_agent_action = state.get("youtube_agent_action")
    company = state.get("company", "Unknown Company")
    
    if not youtube_agent_action:
        print("[YouTube Agent] ERROR: No action specified")
        return {**state, "goto": "meta_agent", "youtube_status": "ERROR", "error": "No YouTube action specified"}
    
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key:
        print("[YouTube Agent] ERROR: YouTube API key missing")
        return {**state, "goto": "meta_agent", "youtube_status": "ERROR", "error": "YouTube API key not configured"}
    
    # Configure YouTube Tool
    config = {
        "youtube_api_key": youtube_api_key,
        "retry_limit": RETRY_LIMIT,
        "multiplier": MULTIPLIER,
        "min_wait": MIN_WAIT,
        "max_wait": MAX_WAIT
    }
    
    try:
        youtube_tool = YoutubeTool(config)
        
        # SEARCH ACTION
        if youtube_agent_action == "search":
            search_queries = state.get("search_queries", [])
            
            if not search_queries or not isinstance(search_queries, list):
                if isinstance(search_queries, str):
                    search_queries = [search_queries]
                else:
                    print("[YouTube Agent] ERROR: No valid search queries provided")
                    return {**state, "goto": "meta_agent", "youtube_status": "ERROR", "error": "No valid search queries provided"}
            
            print(f"[YouTube Agent] Processing {len(search_queries)} search queries")
            
            search_results = {}
            
            for i, query in enumerate(search_queries):
                print(f"[YouTube Agent] Searching for query {i+1}/{len(search_queries)}: {query}")
                
                try:
                    # Use the tool directly without any additional processing
                    results = youtube_tool.search_videos(query=query, max_results=10, order="relevance")
                    
                    search_results[query] = results
                    print(f"[YouTube Agent] Found {len(results)} results for query: {query}")
                    
                    # Add a small delay between queries to avoid rate limiting
                    if i < len(search_queries) - 1:
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"[YouTube Agent] Error searching for query '{query}': {e}")
                    search_results[query] = {"error": str(e)}
            
            state["search_results"] = search_results
            state["youtube_status"] = "DONE"
            
            print(f"[YouTube Agent] Search complete. Processed {len(search_results)} queries.")
        
        # TRANSCRIBE ACTION
        elif youtube_agent_action == "transcribe":
            video_ids = state.get("video_ids", [])
            
            if not video_ids:
                print("[YouTube Agent] ERROR: No video IDs provided")
                return {**state, "goto": "meta_agent", "youtube_status": "ERROR", "error": "No video IDs provided"}
            
            print(f"[YouTube Agent] Transcribing {len(video_ids)} videos")
            
            transcript_results = []
            
            for i, video_info in enumerate(video_ids):
                video_id = video_info.get("id")
                video_title = video_info.get("title", "Unknown Title")
                
                if not video_id:
                    print(f"[YouTube Agent] Error: Missing video ID for entry {i+1}")
                    transcript_results.append({
                        "title": video_title,
                        "id": "unknown",
                        "transcript": "Error: Missing video ID",
                        "error": "Missing video ID"
                    })
                    continue
                
                print(f"[YouTube Agent] Transcribing video {i+1}/{len(video_ids)}: {video_title} (ID: {video_id})")
                
                try:
                    # Use the tool directly to get transcript
                    transcript = youtube_tool.get_transcript(video_id)
                    
                    transcript_data = {
                        "title": video_title,
                        "id": video_id,
                        "transcript": transcript
                    }
                    
                    # NEW: Add transcript summarization with corporate governance focus
                    transcript_data = summarize_transcript(company, transcript_data)
                    transcript_results.append(transcript_data)
                    
                    print(f"[YouTube Agent] Successfully transcribed and summarized video: {video_title}")
                    
                    # Add a small delay between transcription requests
                    if i < len(video_ids) - 1:
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"[YouTube Agent] Error transcribing video {video_id}: {e}")
                    transcript_results.append({
                        "title": video_title,
                        "id": video_id,
                        "transcript": "Error: Failed to retrieve transcript",
                        "transcript_summary": "Error: Failed to retrieve and summarize transcript",
                        "error": str(e)
                    })
            
            state["transcript_results"] = transcript_results
            state["youtube_status"] = "DONE"
            
            print(f"[YouTube Agent] Transcription and summarization complete. Processed {len(transcript_results)} videos.")
        
        # INVALID ACTION
        else:
            print(f"[YouTube Agent] ERROR: Invalid action '{youtube_agent_action}'")
            return {**state, "goto": "meta_agent", "youtube_status": "ERROR", "error": f"Invalid YouTube action: {youtube_agent_action}"}
    
    except Exception as e:
        print(f"[YouTube Agent] Error initializing YouTube tool: {e}")
        return {**state, "goto": "meta_agent", "youtube_status": "ERROR", "error": f"Failed to initialize YouTube tool: {str(e)}"}
    
    return {**state, "goto": "meta_agent"}