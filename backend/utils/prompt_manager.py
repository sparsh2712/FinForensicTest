import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages prompts for different agents in the system.
    Supports a directory-based organization where each agent has its own directory of templates.
    Includes built-in fallback templates for critical operations.
    """
    
    def __init__(self, prompt_dir: Optional[str] = None):
        """
        Initialize the prompt manager.
        """
        # Find prompt directory
        self.prompt_dir = prompt_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
        
        logger.info(f"Initializing PromptManager with prompt_dir: {self.prompt_dir}")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(self.prompt_dir):
            try:
                os.makedirs(self.prompt_dir, exist_ok=True)
                logger.info(f"Created prompt directory: {self.prompt_dir}")
            except Exception as e:
                logger.error(f"Error creating prompt directory {self.prompt_dir}: {e}")
        
        # Setup Jinja environment
        try:
            self.env = Environment(
                loader=FileSystemLoader([self.prompt_dir]),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            logger.info("Jinja2 environment initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Jinja2 environment: {e}")
            self.env = None
        
        # Cache for loaded templates
        self.templates = {}
        
        # Default templates for critical operations
        self.default_templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict:
        """
        Load default templates for critical operations.
        These will be used as fallbacks when the template files don't exist.
        """
        defaults = {
            "research_agent": {
                "generate_queries": {
                    "system": "You are an expert search query generator focused on investigating companies. Generate targeted search queries based on research plans.",
                    "human": """
I need to generate search queries to investigate {{ company }} ({{ industry }} industry).

Research Plan: {{ research_plan }}

{% if query_history %}Previous Queries: {{ query_history }}{% endif %}

Generate a JSON object with categories as keys and arrays of search queries as values. Focus on:
1. Specific issues mentioned in the research plan
2. General queries about financial performance, legal issues, and regulatory actions
3. Different time periods and related entities

Format:
{
  "category1": ["query1", "query2"],
  "category2": ["query1", "query2"]
}
"""
                }
            },
            "meta_agent": {
                "initialize": {
                    "system": "You are a strategic research coordinator focused on financial forensics. You guide the investigation and analysis process.",
                    "human": """
I need a preliminary research plan to investigate {{ company }} in the {{ industry }} industry.

Generate a research plan that covers:
1. Key objectives of the investigation
2. Areas of focus (financial, legal, regulatory, etc.)
3. Important stakeholders to research
4. Specific risk factors to examine

Format your response as JSON with clear structure.
"""
                }
            },
            "analyst_agent": {
                "analyze": {
                    "system": "You are a financial and news analyst specialized in identifying potential issues and risks for companies.",
                    "human": """
Analyze the following research results for {{ company }} in the {{ industry }} industry:

Events found: {{ research_results | tojson }}

For each event:
1. Assess credibility of sources
2. Identify potential impact on the company
3. Note any regulatory, legal, or ethical implications
4. Evaluate timeframe and urgency

Provide a structured analysis for each event.
"""
                }
            }
        }
        
        logger.info(f"Loaded {sum(len(agent) for agent in defaults.values())} default templates")
        return defaults
    
    def load_template(self, agent_name: str, operation: str) -> Optional[Tuple[str, str]]:
        """
        Load a template from file. Supports both directory-based and flat structures.
        Falls back to default templates if the files don't exist.
        
        Args:
            agent_name: Name of the agent (e.g., 'meta_agent')
            operation: Specific operation (e.g., 'system', 'analysis', 'synthesis')
            
        Returns:
            Tuple of (human_template, system_template) or None if not found and no default exists
        """
        template_key = f"{agent_name}/{operation}"
        
        # Return from cache if already loaded
        if template_key in self.templates:
            logger.debug(f"Using cached template for {template_key}")
            return self.templates[template_key]
        
        # Try loading from files
        try:
            agent_dir = os.path.join(self.prompt_dir, agent_name)
            
            # Check if agent directory exists
            if os.path.exists(agent_dir) and os.path.isdir(agent_dir):
                human_template_path = os.path.join(agent_dir, f"{operation}_human.j2")
                system_template_path = os.path.join(agent_dir, f"{operation}_system.j2")
                
                # Check if both files exist
                if os.path.exists(human_template_path) and os.path.exists(system_template_path):
                    with open(human_template_path, 'r', encoding='utf-8') as f:
                        human_template = f.read()
                    with open(system_template_path, 'r', encoding='utf-8') as f:
                        system_template = f.read()
                    
                    self.templates[template_key] = (human_template, system_template)
                    logger.debug(f"Loaded template from files: {template_key}")
                    return human_template, system_template
        
        except Exception as e:
            logger.warning(f"Error loading template files for {template_key}: {e}")
        
        # Fall back to default templates
        if agent_name in self.default_templates and operation in self.default_templates[agent_name]:
            logger.info(f"Using default template for {template_key}")
            default = self.default_templates[agent_name][operation]
            self.templates[template_key] = (default["human"], default["system"])
            return default["human"], default["system"]
        
        # No template found
        logger.warning(f"No template found for {template_key} and no default available")
        return None
    
    def render_template(self, agent_name: str, operation: str, variables: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Render a template with the given variables.
        
        Args:
            agent_name: Name of the agent (e.g., 'meta_agent')
            operation: Specific operation (e.g., 'system', 'analysis')
            variables: Variables to use in the template
            
        Returns:
            Tuple of (human_prompt, system_prompt) or generic prompts if template not found
        """
        variables = variables or {}
        
        try:
            templates = self.load_template(agent_name, operation)
            
            if templates is None:
                # Use generic fallback templates
                logger.warning(f"Using generic fallback templates for {agent_name}/{operation}")
                system_prompt = f"You are a helpful AI assistant working as a {agent_name}."
                human_prompt = f"Analyze the information about {variables.get('company', 'this company')}."
                return system_prompt, human_prompt
            
            human_template, system_template = templates
            
            # Custom filter for JSON conversion
            def tojson(obj):
                return json.dumps(obj, indent=2)
            
            # Create Jinja2 environment for rendering
            env = Environment(autoescape=select_autoescape(['html', 'xml']))
            env.filters['tojson'] = tojson
            
            # Render templates
            human_prompt = env.from_string(human_template).render(**variables)
            system_prompt = env.from_string(system_template).render(**variables)
            
            return system_prompt, human_prompt
            
        except Exception as e:
            logger.error(f"Error rendering template {agent_name}/{operation}: {e}", exc_info=True)
            # Provide a basic fallback prompt
            system_prompt = f"You are a helpful AI assistant working as a {agent_name}."
            human_prompt = f"Analyze the information about {variables.get('company', 'this company')}."
            return system_prompt, human_prompt
    
    def get_prompt(self, agent_name: str, operation: str, variables: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Get a formatted prompt.
        
        Args:
            agent_name: Name of the agent (e.g., 'meta_agent')
            operation: Specific operation (e.g., 'system', 'analysis')
            variables: Variables to use in the template
            
        Returns:
            Tuple of (system_prompt, human_prompt)
        """
        try:
            return self.render_template(agent_name, operation, variables)
        except Exception as e:
            logger.error(f"Error getting prompt for {agent_name}/{operation}: {e}", exc_info=True)
            # Provide a basic fallback prompt
            system_prompt = f"You are a helpful AI assistant working as a {agent_name}."
            human_prompt = f"Analyze the information about {variables.get('company', 'this company')}."
            return system_prompt, human_prompt
    
    def create_template(self, agent_name: str, operation: str, system_content: str, human_content: str) -> None:
        """
        Create or update template files.
        
        Args:
            agent_name: Name of the agent (e.g., 'meta_agent')
            operation: Specific operation (e.g., 'system', 'analysis')
            system_content: System template content
            human_content: Human template content
        """
        # Create agent directory if needed
        agent_dir = os.path.join(self.prompt_dir, agent_name)
        if not os.path.exists(agent_dir):
            try:
                os.makedirs(agent_dir, exist_ok=True)
                logger.info(f"Created agent directory: {agent_dir}")
            except Exception as e:
                logger.error(f"Error creating agent directory {agent_dir}: {e}")
                return
        
        # Create template files in agent directory
        system_template_path = os.path.join(agent_dir, f"{operation}_system.j2")
        human_template_path = os.path.join(agent_dir, f"{operation}_human.j2")
        
        try:
            with open(system_template_path, 'w', encoding='utf-8') as f:
                f.write(system_content)
            with open(human_template_path, 'w', encoding='utf-8') as f:
                f.write(human_content)
                
            logger.info(f"Created template files: {system_template_path} and {human_template_path}")
            
            # Clear cache for this template
            template_key = f"{agent_name}/{operation}"
            if template_key in self.templates:
                del self.templates[template_key]
                
        except Exception as e:
            logger.error(f"Error creating template files: {e}")
    
    def create_default_templates(self) -> None:
        """
        Create all default templates if they don't exist.
        This is useful for initializing a new system.
        """
        for agent_name, operations in self.default_templates.items():
            for operation, templates in operations.items():
                try:
                    # Check if template already exists
                    agent_dir = os.path.join(self.prompt_dir, agent_name)
                    system_path = os.path.join(agent_dir, f"{operation}_system.j2")
                    human_path = os.path.join(agent_dir, f"{operation}_human.j2")
                    
                    if not os.path.exists(system_path) or not os.path.exists(human_path):
                        self.create_template(
                            agent_name=agent_name,
                            operation=operation,
                            system_content=templates["system"],
                            human_content=templates["human"]
                        )
                        logger.info(f"Created default template for {agent_name}/{operation}")
                        
                except Exception as e:
                    logger.error(f"Error creating default template for {agent_name}/{operation}: {e}")