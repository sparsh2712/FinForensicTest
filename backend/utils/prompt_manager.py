# backend/utils/prompt_manager.py
import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages prompts for different agents in the system.
    Supports a directory-based organization where each agent has its own directory of templates.
    """
    
    def __init__(self, prompt_dir: Optional[str] = None):
        """
        Initialize the prompt manager.
        """
        # Find prompt directory
        current_file_path = os.path.abspath(__file__)
        self.prompt_dir = prompt_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","prompts")
        
        if not os.path.exists(self.prompt_dir):
            raise FileNotFoundError(f"Prompt directory not found: {self.prompt_dir}")
        
        # Setup Jinja environment
        self.env = Environment(
            loader=FileSystemLoader([self.prompt_dir]),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Cache for loaded templates
        self.templates = {}
        logger.info(f"PromptManager initialized with template directory: {self.prompt_dir}")
    
    def load_template(self, agent_name: str, operation: str = "system") -> Optional[Template]:
        """
        Load a template from file. Supports both directory-based and flat structures.
        
        Args:
            agent_name: Name of the agent (e.g., 'meta_agent')
            operation: Specific operation (e.g., 'system', 'analysis', 'synthesis')
            
        Returns:
            Jinja2 Template object or None if not found
        """
        template_key = f"{agent_name}/{operation}"
        
        if template_key in self.templates:
            return self.templates[template_key]
        
        agent_dir = os.path.join(self.prompt_dir, agent_name)
        if os.path.exists(agent_dir) and os.path.isdir(agent_dir):
            human_template_file_name = f"{operation}_human.j2"
            system_template_file_name = f"{operation}_system.j2"
            try:
                human_template_path = os.path.join(agent_dir, human_template_file_name)
                system_template_path = os.path.join(agent_dir, system_template_file_name)
                human_template = self.env.get_template(human_template_path)
                system_template = self.env.get_template(system_template_path)
                self.templates[template_key] = (human_template, system_template)
                logger.debug(f"Loaded template: {human_template_path, system_template_path}")
                return (human_template, system_template)
            except Exception as e:
                return None
    
    def render_template(self, agent_name: str, operation: str = "system", variables: Dict[str, Any] = None) -> str:
        """
        Render a template with the given variables.
        
        Args:
            agent_name: Name of the agent (e.g., 'meta_agent')
            operation: Specific operation (e.g., 'system', 'analysis')
            variables: Variables to use in the template
            
        Returns:
            Rendered template string
        """
        variables = variables or {}
        human_template, system_template = self.load_template(agent_name, operation)
        
        if system_template:
            try:
                return (system_template.render(**variables), human_template.render(**variables))
            except Exception as e:
                logger.error(f"Error rendering template {agent_name}/{operation}: {e}")
                return None
    
    def get_prompt(self, agent_name: str, operation: str = "system", variables: Dict[str, Any] = None) -> str:
        """
        Get a formatted prompt.
        
        Args:
            agent_name: Name of the agent (e.g., 'meta_agent')
            operation: Specific operation (e.g., 'system', 'analysis')
            variables: Variables to use in the template
            
        Returns:
            Formatted prompt string
        """
        return self.render_template(agent_name, operation, variables)
    
    def create_template(self, agent_name: str, operation: str, content: str) -> None:
        """
        Create or update a template file.
        
        Args:
            agent_name: Name of the agent (e.g., 'meta_agent')
            operation: Specific operation (e.g., 'system', 'analysis')
            content: Template content
        """
        # Create agent directory if needed
        agent_dir = os.path.join(self.prompt_dir, agent_name)
        if not os.path.exists(agent_dir):
            try:
                os.makedirs(agent_dir, exist_ok=True)
                logger.info(f"Created agent directory: {agent_dir}")
            except Exception as e:
                logger.error(f"Error creating agent directory {agent_dir}: {e}")
                # Fall back to flat structure
                template_path = os.path.join(self.prompt_dir, f"{agent_name}_{operation}.j2")
                try:
                    with open(template_path, 'w') as f:
                        f.write(content)
                    logger.info(f"Created template file (flat structure): {template_path}")
                    return
                except Exception as e2:
                    logger.error(f"Error creating template file {template_path}: {e2}")
                    return
        
        # Create template file in agent directory
        template_path = os.path.join(agent_dir, f"{operation}.j2")
        try:
            with open(template_path, 'w') as f:
                f.write(content)
            logger.info(f"Created template file: {template_path}")
            
            # Clear cache for this template
            template_key = f"{agent_name}/{operation}"
            if template_key in self.templates:
                del self.templates[template_key]
        except Exception as e:
            logger.error(f"Error creating template file {template_path}: {e}")
    
    def create_template_if_not_exists(self, agent_name: str, operation: str, content: str) -> None:
        """
        Create a template file if it doesn't exist.
        
        Args:
            agent_name: Name of the agent (e.g., 'meta_agent')
            operation: Specific operation (e.g., 'system', 'analysis')
            content: Template content
        """
        # Check if template exists in directory structure
        agent_dir = os.path.join(self.prompt_dir, agent_name)
        template_path = os.path.join(agent_dir, f"{operation}.j2")
        
        if os.path.exists(template_path):
            return
        
        # Check if template exists in flat structure
        flat_template_path = os.path.join(self.prompt_dir, f"{agent_name}_{operation}.j2")
        if os.path.exists(flat_template_path):
            return
        
        # Create the template
        self.create_template(agent_name, operation, content)