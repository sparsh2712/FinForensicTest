# FinForensic System

A multi-agent system for conducting forensic analysis of companies, designed to uncover potential issues, controversies, and risks through comprehensive news research and analysis

## =� Overview

FinForensic is an advanced research tool that leverages multiple AI agents working together to:

1. **Research** companies for negative events, regulatory issues, legal problems, and financial irregularities
2. **Analyze** the findings to identify patterns, assess severity, and evaluate potential impacts
3. **Generate** comprehensive reports highlighting key risks and concerns

The system is built using LangGraph for agent orchestration and Streamlit for a user-friendly web interface.

## ( Features

- **Multi-Agent Architecture**:
  - **Meta Agent**: Orchestrates the workflow, evaluates research quality, and directs other agents
  - **Research Agent**: Conducts targeted searches for company information using specialized queries
  - **Analyst Agent**: Evaluates collected data to identify patterns and assess risks

- **Comprehensive Research**: 
  - Focuses on negative events, controversies, and potential red flags
  - Covers multiple areas including legal issues, financial concerns, and operational problems
  
- **Intelligent Analysis**:
  - Quality assessment of research findings
  - Pattern recognition across multiple events
  - Risk evaluation and severity assessment
  
- **Interactive UI**:
  - Real-time progress monitoring
  - Agent status visualization
  - Downloadable PDF reports

## =� Getting Started

### Prerequisites

- Python 3.12 or higher
- API keys for:
  - Google Gemini or other supported LLM providers
  - SerpAPI for search queries

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FinForensicTest.git
   cd FinForensicTest
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   GOOGLE_API_KEY=your_google_gemini_api_key
   SERPAPI_API_KEY=your_serpapi_key
   ```

### Running the Application

Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The application will be available at http://localhost:8501

## =� Usage

1. Enter a company name in the sidebar
2. Optionally specify the industry for more focused research
3. Click "Start Analysis" to begin the process
4. Monitor the analysis progress in real-time
5. View and download the final report when the analysis completes

## =� Project Structure

```
FinForensicTest/
   backend/               # Core system components
      agents/            # Agent implementations
      core/              # Main system architecture
      prompts/           # Agent prompts and configurations
      utils/             # Utility functions
   debug/                 # Debug information and checkpoints
   markdowns/             # Generated reports
   pyproject.toml         # Project configuration
   requirements.txt       # Dependencies
   streamlit_app.py       # Web interface
```

## >� System Architecture

The system uses a directed graph workflow managed by LangGraph:

1. **Research Phase**: 
   - Research agent collects information about potential issues
   - Meta agent evaluates quality and completeness
   - Multiple research iterations until quality threshold is met

2. **Analysis Phase**:
   - Analyst agent evaluates the collected information
   - Identifies patterns and severity
   - Categorizes findings and assesses risks

3. **Reporting Phase**:
   - Final report generation with structured sections
   - PDF conversion for sharing and archiving
   - Quality assurance to ensure completeness

## =� Technologies Used

- **Core Framework**:
  - LangGraph for agent workflow management
  - LangChain for LLM interactions
  - Streamlit for web interface

- **AI/LLM**:
  - Google Gemini (via langchain_google_genai)
  - Also supports Anthropic and OpenAI

- **Search & Processing**:
  - SerpAPI for search queries
  - Markdown processing
  - WeasyPrint for PDF conversion

## >� Future Improvements

- Additional data sources beyond web search
- Enhanced pattern recognition across financial data
- Integration with regulatory databases
- Improved report customization options
- Expanded agent capabilities for deeper analysis

## =� License

[Your License Information]

## > Contributing

Contributions, issues, and feature requests are welcome!

## =O Acknowledgements

- [List any acknowledgements or credits here]