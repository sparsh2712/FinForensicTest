from backend.utils.prompt_manager import PromptManager

prompt_manager = PromptManager("/Users/sparsh/Desktop/FinForensicTest/backend/prompts")

variables = {
            "company": "company",
            "current_date": "current_date",
            "rag_json": "rag_json",
            "transcript_json": "transcript_json",
            "corporate_json": "corporate_json"
        }
        
system_prompt, human_prompt = prompt_manager.get_prompt(
    "corporate_meta_writer_agent", 
    "generate_final_report", 
    variables
)

print(human_prompt)