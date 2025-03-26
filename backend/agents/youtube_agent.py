import json
import time
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import googleapiclient.errors
from tenacity import retry, stop_after_attempt, wait_exponential
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import os

logger = logging.getLogger("youtube_agent")

class YoutubeTool:    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("youtube_agent")
        self.youtube_api_key = config.get("youtube_api_key")
        self.retry_limit = config.get("retry_limit", RETRY_LIMIT)
        self.multiplier = config.get("multiplier", MULTIPLIER)
        self.min_wait = config.get("min_wait", MIN_WAIT)
        self.max_wait = config.get("max_wait", MAX_WAIT)
        
        self.youtube = googleapiclient.discovery.build(
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
            transcript = self.transcriptor.get_transcript(video_id)
            
            self.logger.info(f"Successfully fetched transcript for video: {video_id}")
            
            return " ".join(snippet.text for snippet in transcript)
        except Exception as e:
            self.logger.error(f"Error fetching transcript: {e}")
            return "Failed to transcribe the video"
    
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
