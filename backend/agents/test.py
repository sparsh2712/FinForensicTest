from backend.utils.prompt_manager import PromptManager

prompt_manager = PromptManager("/Users/sparsh/Desktop/FinForensicTest/backend/prompts")

variables = {
            "company": "company",
            "title": "title",
            "event_name": "event_name",
            "content": "content"
        }

system_prompt, human_prompt  = prompt_manager.get_prompt("analyst_agent", "forensic_insights_extract", variables)

print(system_prompt)