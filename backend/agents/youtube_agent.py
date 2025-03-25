import json
import time
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import os

logger = logging.getLogger("youtube_agent")

class YouTubeToolSimple:
    """Simplified YouTube tool for fetching videos and transcripts"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        if not self.api_key:
            logger.warning("YouTube API key not provided, using limited functionality")
        
        try:
            if self.api_key:
                self.youtube = build("youtube", "v3", developerKey=self.api_key)
            else:
                self.youtube = None
        except Exception as e:
            logger.error(f"Error initializing YouTube API: {str(e)}")
            self.youtube = None
    
    def search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for videos related to a query"""
        if not self.youtube:
            logger.warning("YouTube API not initialized, returning mock data")
            return self._get_mock_videos(query, max_results)
            
        try:
            search_response = self.youtube.search().list(
                q=query,
                part="snippet",
                maxResults=max_results,
                type="video"
            ).execute()
            
            videos = []
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]
                video = {
                    "id": video_id,
                    "title": item["snippet"]["title"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "description": item["snippet"]["description"],
                    "published_at": item["snippet"]["publishedAt"]
                }
                videos.append(video)
            
            return videos
        except Exception as e:
            logger.error(f"Error searching YouTube videos: {str(e)}")
            return self._get_mock_videos(query, max_results)
    
    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript for a video"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([item["text"] for item in transcript_list])
            return transcript
        except Exception as e:
            logger.error(f"Error getting transcript for video {video_id}: {str(e)}")
            return None
    
    def get_channel_id(self, channel_name: str) -> Optional[str]:
        """Get channel ID from channel name"""
        if not self.youtube:
            return None
            
        try:
            search_response = self.youtube.search().list(
                q=channel_name,
                part="snippet",
                maxResults=1,
                type="channel"
            ).execute()
            
            if search_response.get("items"):
                return search_response["items"][0]["id"]["channelId"]
            return None
        except Exception as e:
            logger.error(f"Error getting channel ID: {str(e)}")
            return None
    
    def get_videos_from_channel(self, channel_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get videos from a specific channel"""
        if not self.youtube:
            return []
            
        try:
            search_response = self.youtube.search().list(
                channelId=channel_id,
                part="snippet",
                maxResults=max_results,
                type="video",
                order="date"
            ).execute()
            
            videos = []
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]
                video = {
                    "id": video_id,
                    "title": item["snippet"]["title"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "description": item["snippet"]["description"],
                    "published_at": item["snippet"]["publishedAt"]
                }
                videos.append(video)
            
            return videos
        except Exception as e:
            logger.error(f"Error getting videos from channel: {str(e)}")
            return []
    
    def _get_mock_videos(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Get mock videos for when API is not available"""
        company_name = query.split()[0]  # Assume first word is company name
        videos = []
        
        for i in range(min(max_results, 5)):
            video_id = f"mock{i}video{hash(query) % 1000}"
            video = {
                "id": video_id,
                "title": f"{company_name} {['Quarterly Results', 'Product Launch', 'CEO Interview', 'Industry Analysis', 'Financial Overview'][i % 5]}",
                "channel_title": f"{['Bloomberg', 'CNBC', 'Financial Times', 'Business Insider', 'Wall Street Journal'][i % 5]}",
                "description": f"This video discusses {company_name}'s recent developments and market position.",
                "published_at": datetime.now().isoformat()
            }
            videos.append(video)
        
        return videos

def analyze_transcript(transcript: str, company: str) -> Dict[str, Any]:
    """Analyze a transcript for forensically-relevant content"""
    if not transcript:
        return {
            "forensic_relevance": "unknown",
            "red_flags": [],
            "summary": "No transcript available for analysis"
        }
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
        system_prompt = """
        You are a forensic financial analyst specializing in detecting potential issues in corporate communications.
        Analyze this video transcript and extract any information that could be forensically relevant.
        Focus on:
        
        1. Statements that might indicate financial irregularities
        2. Discussions of legal issues or regulatory matters
        3. Disclosure of operational problems
        4. Leadership or governance concerns
        5. Contradictions to official company statements
        
        Format your response as a JSON object with the following fields:
        - forensic_relevance: "high", "medium", "low", or "unknown"
        - red_flags: [List of specific concerns identified]
        - summary: A brief assessment of the content
        """
        
        human_prompt = f"""
        Company: {company}
        
        Video Transcript:
        {transcript[:5000]}... (transcript truncated for processing)
        
        Analyze this transcript for forensically-relevant information about {company}.
        """
        
        messages = [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
        
        response = llm.invoke(messages)
        response_content = response.content.strip()
        
        if "```json" in response_content:
            json_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_content = response_content.split("```")[1].strip()
        else:
            json_content = response_content
            
        analysis = json.loads(json_content)
        return analysis
    
    except Exception as e:
        logger.error(f"Error analyzing transcript: {str(e)}")
        return {
            "forensic_relevance": "unknown",
            "red_flags": ["Analysis failed: " + str(e)],
            "summary": "Error during transcript analysis"
        }

def youtube_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified YouTube agent that analyzes videos related to a company.
    
    Args:
        state: The current state dictionary containing:
            - company: Company name
            - youtube_queries: Optional list of custom search queries
            - youtube_channels: Optional list of channel names to analyze
    
    Returns:
        Updated state containing YouTube analysis results and next routing information
    """
    logger.info(f"Starting YouTube agent for {state.get('company')}")
    
    try:
        company = state.get("company", "")
        industry = state.get("industry", "")
        
        if not company:
            logger.error("Company name is missing!")
            return {**state, "goto": "meta_agent", "youtube_status": "ERROR", "error": "Company name is missing"}
        
        # Initialize the YouTube tool
        youtube_tool = YouTubeToolSimple()
        
        # Get custom queries or generate default ones
        youtube_queries = state.get("youtube_queries", [])
        if not youtube_queries:
            youtube_queries = [
                f"{company}",
                f"{company} {industry}",
                f"{company} financial issues",
                f"{company} controversy",
                f"{company} regulatory issues"
            ]
        
        # Get custom channels to analyze
        youtube_channels = state.get("youtube_channels", [])
        
        logger.info(f"Analyzing {len(youtube_queries)} queries and {len(youtube_channels)} channels for {company}")
        
        # Search for videos based on queries
        all_videos = []
        for query in youtube_queries:
            try:
                videos = youtube_tool.search_videos(query, max_results=5)
                all_videos.extend(videos)
                logger.info(f"Found {len(videos)} videos for query: {query}")
            except Exception as e:
                logger.error(f"Error searching for '{query}': {str(e)}")
        
        # Deduplicate videos
        unique_videos = {}
        for video in all_videos:
            video_id = video.get("id")
            if video_id and video_id not in unique_videos:
                unique_videos[video_id] = video
        
        # Get videos from specific channels if provided
        for channel_name in youtube_channels:
            try:
                channel_id = youtube_tool.get_channel_id(channel_name)
                if channel_id:
                    channel_videos = youtube_tool.get_videos_from_channel(channel_id, max_results=5)
                    for video in channel_videos:
                        video_id = video.get("id")
                        if video_id and video_id not in unique_videos:
                            unique_videos[video_id] = video
                    logger.info(f"Found {len(channel_videos)} videos from channel: {channel_name}")
            except Exception as e:
                logger.error(f"Error getting videos from channel '{channel_name}': {str(e)}")
        
        logger.info(f"Processing {len(unique_videos)} unique videos")
        
        # Process top 5 videos
        processed_videos = []
        for i, (video_id, video) in enumerate(list(unique_videos.items())[:5]):
            logger.info(f"Processing video {i+1}/{min(5, len(unique_videos))}: {video.get('title')}")
            
            try:
                # Get transcript
                transcript = youtube_tool.get_transcript(video_id)
                
                # Analyze transcript if available
                analysis = None
                if transcript:
                    analysis = analyze_transcript(transcript, company)
                
                # Add to processed videos
                processed_video = {
                    "video_id": video_id,
                    "title": video.get("title", ""),
                    "channel": video.get("channel_title", ""),
                    "description": video.get("description", ""),
                    "has_transcript": transcript is not None,
                    "transcript_length": len(transcript) if transcript else 0,
                    "forensic_relevance": analysis.get("forensic_relevance", "unknown") if analysis else "unknown",
                    "red_flags": analysis.get("red_flags", []) if analysis else [],
                    "summary": analysis.get("summary", "") if analysis else "No analysis available"
                }
                processed_videos.append(processed_video)
                
            except Exception as e:
                logger.error(f"Error processing video {video_id}: {str(e)}")
        
        # Generate an overall summary
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            
            system_prompt = """
            You are a forensic financial analyst summarizing YouTube content related to a company.
            Based on the analysis of multiple videos, provide a comprehensive summary of findings.
            
            Format your response as a JSON object with the following fields:
            - overall_assessment: Brief assessment of the company based on video content
            - key_insights: List of the most important insights
            - red_flags: List of potential issues identified
            - notable_videos: List of the most significant videos analyzed
            """
            
            human_prompt = f"""
            Company: {company}
            
            Videos Analyzed:
            {json.dumps(processed_videos, indent=2)}
            
            Provide a comprehensive summary of the YouTube content related to {company}.
            """
            
            messages = [
                ("system", system_prompt),
                ("human", human_prompt)
            ]
            
            response = llm.invoke(messages)
            response_content = response.content.strip()
            
            if "```json" in response_content:
                json_content = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_content = response_content.split("```")[1].strip()
            else:
                json_content = response_content
                
            summary = json.loads(json_content)
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            summary = {
                "overall_assessment": "Could not generate assessment due to technical error",
                "key_insights": ["Analysis incomplete due to technical issues"],
                "red_flags": ["Summary generation failed"],
                "notable_videos": []
            }
        
        # Prepare the results
        youtube_results = {
            "videos": processed_videos,
            "summary": summary,
            "red_flags": summary.get("red_flags", []),
            "queries": youtube_queries,
            "channels": youtube_channels
        }
        
        # Update state with results
        state["youtube_results"] = youtube_results
        state["youtube_status"] = "DONE"
        
        # If synchronous_pipeline is set, use the next_agent value, otherwise go to meta_agent
        goto = "meta_agent"
        if state.get("synchronous_pipeline", False):
            goto = state.get("next_agent", "meta_agent")
        
        logger.info(f"YouTube agent completed successfully for {company}")
        return {**state, "goto": goto}
    
    except Exception as e:
        logger.error(f"Error in YouTube agent: {str(e)}")
        return {
            **state,
            "goto": "meta_agent",
            "youtube_status": "ERROR",
            "error": f"Error in YouTube agent: {str(e)}"
        }