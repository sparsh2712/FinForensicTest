import streamlit as st
import time
import os
import base64
import re
from datetime import datetime
import tempfile
from pathlib import Path
import logging
import sys
import traceback # Import traceback for detailed error logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit_app.log"), # Log to file
        logging.StreamHandler(sys.stdout)      # Log to console
    ]
)
logger = logging.getLogger(__name__) # Use __name__ for logger

# --- Import Backend Modules with Error Handling ---
try:
    # Import the config manager first
    from backend.utils.config_manager import ConfigManager
    config_manager = ConfigManager()
    logger.info("Successfully imported and initialized ConfigManager.")

    # Import system types
    from backend.core.news_forensic import NewsForensicSystem
    from backend.core.corporate_governance_workflow import CorporateGovernanceSystem
    from backend.utils.pdf_generator import convert_markdown_to_pdf
    logger.info("Successfully imported core backend modules (NewsForensic, CorpGov, PDFGen).")

except ImportError as e:
    logger.error(f"Fatal Error: Failed to import core backend modules: {e}", exc_info=True)
    # Display error in Streamlit and stop execution if imports fail
    st.error(f"Fatal Error: Failed to load core application modules: {e}. "
             "Please check the backend installation and dependencies. See logs for details.")
    st.stop() # Stop the Streamlit app if critical modules are missing
except Exception as e:
    logger.error(f"Fatal Error: Failed during initial setup: {e}", exc_info=True)
    st.error(f"Fatal Error: An unexpected error occurred during setup: {e}. Check logs for details.")
    st.stop()


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Financial Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# Define default values for session state variables
default_values = {
    "messages": [{"role": "assistant", "content": "Welcome to the Financial Analysis System! Please provide company information and select analysis options."}],
    "report": None,
    "system": None, # Consider removing if not used directly
    "cg_system": None, # Consider removing if not used directly
    "analysis_running": False,
    "step": 1,
    "company_name": "",
    "company_symbol": "",
    "industry": "",
    "company_website": "",
    "include_corporate_governance": False,
    "include_youtube_transcripts": False, # Feature flag, potentially link to config
    "uploaded_pdf": None, # Stores the name of the uploaded file
    "pdf_path": None,     # Stores the temporary path of the saved PDF
    "pdf_queries": [],
    "configuration": {}   # Stores the review configuration
}

# Initialize session state keys if they don't exist
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions ---

def display_pdf(pdf_path):
    """Display PDF file inline using base64 encoding."""
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        # Embed PDF in an iframe
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        logger.info(f"Successfully displayed PDF: {pdf_path}")
    except FileNotFoundError:
        logger.error(f"PDF display error: File not found at {pdf_path}")
        st.error(f"Error: PDF file not found at path: {pdf_path}")
    except Exception as e:
        logger.error(f"Error displaying PDF {pdf_path}: {e}", exc_info=True)
        st.error(f"An error occurred while trying to display the PDF: {e}")

def get_pdf_download_link(pdf_path, filename):
    """Generate a base64 encoded download link for a PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode('utf-8')
        # Create download link HTML
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-btn">Download PDF Report</a>'
        logger.info(f"Generated download link for {filename} from {pdf_path}")
        return href
    except FileNotFoundError:
        logger.error(f"PDF download link error: File not found at {pdf_path}")
        return "<span>Error: Report file not found for download.</span>"
    except Exception as e:
        logger.error(f"Error generating download link for {pdf_path}: {e}", exc_info=True)
        return f"<span>Error generating download link: {e}</span>"

def run_unified_analysis():
    """
    Orchestrates the selected financial analyses (News Forensic, Corporate Governance).
    Logs progress and errors, updates session state messages, and generates a combined report.
    """
    logger.info("Starting unified analysis process.")
    st.session_state.analysis_running = True

    # Retrieve necessary info from session state
    company_name = st.session_state.company_name
    industry = st.session_state.industry
    include_governance = st.session_state.include_corporate_governance
    # include_youtube = st.session_state.include_youtube_transcripts # Flag for future use
    pdf_path = st.session_state.pdf_path # Path to uploaded PDF, if any
    company_symbol = st.session_state.company_symbol

    # Update UI messages
    st.session_state.messages.append({"role": "user", "content": f"Analyze {company_name}"})
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"🚀 Starting comprehensive analysis for {company_name}. Please wait..."
    })

    # Initialize results and report variables
    forensic_result = None
    governance_result = None
    combined_report = None

    try:
        # --- Run News Forensic Analysis ---
        logger.info(f"Beginning News Forensic Analysis for: {company_name}")
        with st.spinner(f"Running news forensic analysis for {company_name}..."):
            try:
                logger.info(f"Initializing NewsForensicSystem for {company_name}")
                news_system = NewsForensicSystem() # Assumes constructor handles its own config/setup

                # Get max iterations from config, default to 6
                max_iterations = config_manager.get_config("max_iterations", 6)
                logger.info(f"Running news forensic analysis with max_iterations={max_iterations}")

                # Execute the analysis
                forensic_result = news_system.run(company_name, industry, max_iterations=max_iterations)
                # Example: forensic_result = {"final_report": "...", "rag_results": {"Q1": "A1"}}

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ News forensic analysis complete for {company_name}."
                })
                logger.info(f"News forensic analysis completed successfully for {company_name}")

            except Exception as news_err:
                # Log detailed error and update UI
                logger.error(f"Error during News Forensic Analysis for {company_name}: {news_err}", exc_info=True)
                st.error(f"Error during News Forensic Analysis: {news_err}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Failed during news forensic analysis for {company_name}. See logs for details."
                })

        # --- Run Corporate Governance Analysis (if selected) ---
        if include_governance and company_symbol:
            logger.info(f"Beginning Corporate Governance Analysis for: {company_name} ({company_symbol})")
            with st.spinner(f"Running corporate governance analysis for {company_name}..."):
                try:
                    logger.info(f"Initializing CorporateGovernanceSystem for {company_name} ({company_symbol})")
                    cg_system = CorporateGovernanceSystem() # Assumes constructor handles setup

                    # Get max iterations from config
                    max_iterations = config_manager.get_config("max_iterations", 6)
                    logger.info(f"Running corporate governance analysis with max_iterations={max_iterations}")

                    # Check if PDF path exists, otherwise pass None
                    pdf_to_use = pdf_path if pdf_path and os.path.exists(pdf_path) else None
                    if not pdf_to_use:
                        logger.warning(f"PDF path {pdf_path} not found or not provided, continuing corporate governance analysis without document analysis")
                        
                    # Execute the analysis with optional PDF path
                    governance_result = cg_system.run(company_name, company_symbol, pdf_to_use, max_iterations=max_iterations)
                    # Example: governance_result = {"final_report": "..."}

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ Corporate governance analysis complete for {company_name}."
                    })
                    logger.info(f"Corporate governance analysis completed successfully for {company_name}")

                except Exception as cg_err:
                    # Log detailed error and update UI
                    logger.error(f"Error during Corporate Governance Analysis for {company_name}: {cg_err}", exc_info=True)
                    st.error(f"Error during Corporate Governance Analysis: {cg_err}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Failed during corporate governance analysis for {company_name}. See logs for details."
                    })
        elif include_governance and not company_symbol:
             logger.warning("Corporate Governance analysis selected but no company symbol provided.")
             st.session_state.messages.append({
                 "role": "assistant",
                 "content": f"⚠️ Corporate Governance analysis skipped: Company symbol (NSE Ticker) is required."
             })

        # --- Handle Custom Document Queries (RAG) ---
        # This assumes RAG might be part of NewsForensicSystem or needs explicit call
        if pdf_path and os.path.exists(pdf_path) and st.session_state.pdf_queries:
            logger.info(f"Processing {len(st.session_state.pdf_queries)} custom queries for document: {pdf_path}")
            with st.spinner(f"Analyzing document with custom queries..."):
                # Placeholder: If RAG is separate, call it here.
                # e.g., rag_results = run_separate_rag(pdf_path, st.session_state.pdf_queries)
                # For now, assume news_system might handle it if configured.
                # Ensure forensic_result exists to potentially store RAG outputs
                if not forensic_result: forensic_result = {}
                if "rag_results" not in forensic_result: forensic_result["rag_results"] = {}

                # Add placeholder/results to forensic_result (actual results depend on backend)
                for query in st.session_state.pdf_queries:
                    if query not in forensic_result.get("rag_results", {}):
                        # This might be populated by the news_system run or a separate RAG call
                        forensic_result["rag_results"][query] = forensic_result.get("rag_results", {}).get(query, "Processing...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"ℹ️ Processed custom document queries (results included if available)."
                })


        # --- Generate Combined Report ---
        logger.info("Generating combined report.")
        report_parts = []

        # Add forensic report section
        if forensic_result and "final_report" in forensic_result and forensic_result["final_report"]:
            report_parts.append("## News Forensic Analysis\n\n" + forensic_result["final_report"])
            logger.info("Added News Forensic Analysis to report.")

        # Add governance report section
        if governance_result and "final_report" in governance_result and governance_result["final_report"]:
            report_parts.append("## Corporate Governance Analysis\n\n" + governance_result["final_report"])
            logger.info("Added Corporate Governance Analysis to report.")

        # Add RAG results section
        if forensic_result and "rag_results" in forensic_result and st.session_state.pdf_queries:
            rag_section_content = []
            for query, result in forensic_result["rag_results"].items():
                rag_section_content.append(f"### Query: {query}\n\n{result or '_No result found_'}\n")
            if rag_section_content:
                report_parts.append("## Document Query Results\n\n" + "\n".join(rag_section_content))
                logger.info("Added Document Query Results to report.")

        # --- Combine and Save Report ---
        if report_parts:
            combined_report = "\n\n---\n\n".join(report_parts) # Use markdown separator
            st.session_state.report = combined_report
            logger.info(f"Combined report generated with {len(report_parts)} sections.")

            # Prepare filenames and directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_company_name = re.sub(r'[\\/*?:"<>|]', "", company_name).replace(' ', '_') # Sanitize filename
            report_filename_base = f"{safe_company_name}_{timestamp}"
            report_filename_md = f"{report_filename_base}.md"
            report_filename_pdf = f"{report_filename_base}.pdf"

            # Get reports directory from config
            try:
                reports_dir = config_manager.get_path("reports_dir", default="reports") # Provide default
                combined_reports_dir = os.path.join(reports_dir, "combined_reports")
            except Exception as config_err:
                 logger.error(f"Error getting reports directory from config: {config_err}", exc_info=True)
                 st.error(f"Configuration error: Could not determine report save directory. Defaulting to './reports/combined_reports'")
                 combined_reports_dir = os.path.join("reports", "combined_reports") # Fallback

            # Ensure directory exists
            os.makedirs(combined_reports_dir, exist_ok=True)
            logger.info(f"Ensured report directory exists: {combined_reports_dir}")

            md_path = os.path.join(combined_reports_dir, report_filename_md)
            pdf_output_path = os.path.join(combined_reports_dir, report_filename_pdf)

            # Save Markdown and Convert to PDF
            try:
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(combined_report)
                logger.info(f"Saved markdown report to: {md_path}")

                # Generate PDF
                logger.info(f"Attempting PDF conversion for: {md_path} -> {pdf_output_path}")
                convert_markdown_to_pdf(md_path, pdf_output_path) # Assumes this function handles errors
                logger.info(f"Successfully saved PDF report to: {pdf_output_path}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ Combined analysis report generated and saved successfully."
                })
            except Exception as save_err:
                # Log error during save/convert and inform user
                logger.error(f"Error saving report (MD or PDF): {save_err}", exc_info=True)
                st.error(f"Error saving or converting report file: {save_err}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Report content generated, but failed during save/PDF conversion. Check logs."
                })
        else:
            # No report content generated
            st.session_state.report = None
            logger.warning("Analysis complete, but no report content was generated.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"ℹ️ Analysis complete. No report content generated based on selected options and available data."
            })

    except Exception as e:
        # Catch-all for unexpected errors during the main analysis flow
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error during unified analysis orchestration: {e}\n{error_trace}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ An critical unexpected error occurred during analysis: {str(e)}. Please check the application logs."
        })
        st.error(f"An critical unexpected error occurred: {e}. See logs for detailed traceback.")

    finally:
        # Ensure analysis running flag is reset and move to results step
        st.session_state.analysis_running = False
        st.session_state.step = 5
        logger.info(f"Unified analysis process finished for {company_name}. Moving to results step.")
        # No st.rerun() here, it's called by the button press that triggers this function


# --- Custom CSS ---
# (Using the well-structured CSS from the previous version)
st.markdown("""
<style>
/* --- General Styles --- */
body {
    color: #E0E0E0; /* Default text color */
}
[data-testid="stAppViewContainer"] {
    background-color: #1E1E1E; /* Dark background */
}
[data-testid="stSidebar"] {
    background-color: #1E1E1E; /* Sidebar background */
}
/* --- Input Labels --- */
.stTextInput label, .stTextArea label, .stSelectbox label,
.stCheckbox label, .stFileUploader label, [data-testid="stRadio"] label {
    color: #E0E0E0 !important; /* Light color for labels */
}
/* --- Buttons --- */
.download-btn {
    display: inline-block; background-color: #2563EB; color: white !important;
    padding: 0.5rem 1rem; text-decoration: none; border-radius: 0.25rem;
    font-weight: 500; margin-top: 1rem; border: none; cursor: pointer;
}
.download-btn:hover { background-color: #1D4ED8; color: white !important; }

.stButton>button {
    background-color: #2563EB; color: white; font-weight: 500; border: none;
    padding: 0.5rem 1rem; border-radius: 0.25rem;
}
.stButton>button:hover { background-color: #1D4ED8; border: none; }
.stButton>button:disabled { background-color: #555; color: #aaa; cursor: not-allowed; }

/* --- Step Indicators --- */
.step-container {
    display: flex; justify-content: space-between; margin-bottom: 2rem;
    padding: 1rem; background-color: #161b22; border-radius: 0.5rem;
}
.step {
    display: flex; flex-direction: column; align-items: center;
    flex: 1; position: relative; color: #c9d1d9;
}
.step:not(:last-child)::after {
    content: ''; position: absolute; top: 1.5rem; right: -50%;
    width: 100%; height: 2px; background-color: #30363d; z-index: 0;
}
.step-number {
    width: 3rem; height: 3rem; border-radius: 50%; background-color: #21262d;
    color: #c9d1d9; display: flex; justify-content: center; align-items: center;
    font-weight: bold; z-index: 1; margin-bottom: 0.5rem; border: 2px solid #30363d;
}
.step-title { font-size: 0.8rem; text-align: center; font-weight: 500; }
.step-active .step-number { background-color: #2563EB; color: white; border-color: #1D4ED8; }
.step-active .step-title { color: white; font-weight: bold; }
.step-completed .step-number { background-color: #059669; color: white; border-color: #047857; }
.step-completed .step-title { color: #8b949e; }

/* --- Report Container (GitHub Dark Style) --- */
.report-container {
    background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 6px; padding: 24px; margin-bottom: 16px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    font-size: 16px; line-height: 1.5; word-wrap: break-word;
}
.report-container h1, .report-container h2, .report-container h3, .report-container h4 { color: #e6edf3 !important; font-weight: 600; }
.report-container h1 { font-size: 2em; padding-bottom: 0.3em; border-bottom: 1px solid #21262d; margin-bottom: 16px; }
.report-container h2 { font-size: 1.5em; padding-bottom: 0.3em; border-bottom: 1px solid #21262d; margin-top: 24px; margin-bottom: 16px; }
.report-container h3 { font-size: 1.25em; margin-top: 24px; margin-bottom: 16px; }
.report-container h4 { font-size: 1em; margin-top: 24px; margin-bottom: 16px; }
.report-container p { margin-top: 0; margin-bottom: 16px; }
.report-container ul, .report-container ol { padding-left: 2em; margin-top: 0; margin-bottom: 16px; }
.report-container li { margin-top: 0.25em; }
.report-container pre {
    padding: 16px; overflow: auto; font-size: 85%; line-height: 1.45;
    background-color: #161b22; border-radius: 6px; margin-top: 0; margin-bottom: 16px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
.report-container :not(pre) > code {
    padding: 0.2em 0.4em; margin: 0; font-size: 85%;
    background-color: rgba(175,184,193,0.2); border-radius: 6px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
.report-container table {
    display: block; width: 100%; overflow: auto; margin: 0 0 16px;
    border-spacing: 0; border-collapse: collapse; border: 1px solid #30363d;
}
.report-container th { padding: 6px 13px; border: 1px solid #30363d; font-weight: 600; background-color: #161b22; color: #e6edf3; }
.report-container td { padding: 6px 13px; border: 1px solid #30363d; color: #c9d1d9; }
.report-container tr { background-color: #0d1117; border-top: 1px solid #21262d; }
.report-container tr:nth-child(2n) { background-color: #161b22; }
.report-container blockquote { padding: 0 1em; color: #8b949e; border-left: 0.25em solid #30363d; margin: 0 0 16px; }
.report-container hr { height: 0.25em; padding: 0; margin: 24px 0; background-color: #30363d; border: 0; }

/* --- Streamlit Tabs Styling --- */
[data-testid="stTabs"] { margin-top: 1rem; }
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 4px 4px 0 0; padding: 0.5rem 1rem; color: #8b949e;
    background-color: #161b22; border: 1px solid #30363d; border-bottom: none;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e6edf3; background-color: #0d1117; border: 1px solid #30363d;
    border-bottom: 1px solid #0d1117; margin-bottom: -1px;
}
/* --- Log Message Styling --- */
.log-message {
    padding: 0.3rem 0.5rem;
    margin-bottom: 0.3rem;
    border-radius: 4px;
    font-size: 0.9em;
    line-height: 1.4;
}
.log-user { background-color: #2a3b4d; } /* Slightly different background for user messages */
.log-assistant { background-color: #161b22; } /* Match step container bg */
.log-error { background-color: #4d2a2a; border-left: 3px solid #f85149; } /* Error highlight */
.log-warning { background-color: #4d4d2a; border-left: 3px solid #f8d149; } /* Warning highlight */
.log-info { background-color: #1f3b3d; border-left: 3px solid #49c5f8; } /* Info highlight */
.log-success { background-color: #1f3d2a; border-left: 3px solid #49f871; } /* Success highlight */

</style>
""", unsafe_allow_html=True)

# --- UI Flow & Step Logic ---

# Define step titles for the indicator UI
step_titles = {
    1: "Company Info",
    2: "Analysis Options",
    3: "Document Upload",
    4: "Review & Start",
    5: "Results"
}

# --- Display Step Indicators ---
st.markdown('<div class="step-container">', unsafe_allow_html=True)
for step_num, step_title in step_titles.items():
    step_class = ""
    step_icon = str(step_num) # Default icon is the number
    if step_num == st.session_state.step:
        step_class = "step-active"
    elif step_num < st.session_state.step:
        step_class = "step-completed"
        step_icon = "✓" # Use checkmark for completed steps

    # Generate HTML for each step indicator
    st.markdown(f'''
    <div class="step {step_class}">
        <div class="step-number">{step_icon}</div>
        <div class="step-title">{step_title}</div>
    </div>
    ''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Main Application Logic based on Current Step ---

# == Step 1: Company Information ==
if st.session_state.step == 1:
    st.header("Step 1: Enter Company Information")
    logger.debug("Displaying Step 1: Company Information")

    # Use a form to batch input updates
    with st.form("company_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            # Use unique keys for widgets to avoid conflicts
            company_name = st.text_input("Company Name*", value=st.session_state.company_name, key="company_name_input_step1")
            industry = st.text_input("Industry", value=st.session_state.industry, key="industry_input_step1")
        with col2:
            company_symbol = st.text_input("NSE Symbol/Ticker", value=st.session_state.company_symbol,
                                           help="Required for corporate governance analysis", key="company_symbol_input_step1")
            company_website = st.text_input("Company Website (Optional)", value=st.session_state.company_website, key="company_website_input_step1")

        st.caption("*Required field")
        submitted = st.form_submit_button("Continue to Analysis Options")

        if submitted:
            logger.info("Company info form submitted.")
            # Validate required fields
            if not company_name:
                st.error("Company Name is required.")
                logger.warning("Company info submission failed: Name is missing.")
            else:
                # Update session state from form inputs using their keys
                st.session_state.company_name = st.session_state.company_name_input_step1
                st.session_state.industry = st.session_state.industry_input_step1
                st.session_state.company_symbol = st.session_state.company_symbol_input_step1
                st.session_state.company_website = st.session_state.company_website_input_step1
                logger.info(f"Updated company info: Name='{st.session_state.company_name}', Symbol='{st.session_state.company_symbol}'")
                # Proceed to the next step
                st.session_state.step = 2
                st.rerun() # Rerun to display the next step

# == Step 2: Analysis Options ==
elif st.session_state.step == 2:
    st.header("Step 2: Select Analysis Options")
    logger.debug("Displaying Step 2: Analysis Options")
    st.subheader("What analysis components would you like to include?")

    # Corporate Governance Checkbox
    include_corp_gov = st.checkbox("Include Corporate Governance Analysis",
                                      value=st.session_state.include_corporate_governance,
                                      key="cb_governance_step2",
                                      help="Requires NSE Symbol/Ticker provided in Step 1.")

    # Display warning if Corp Gov is checked but symbol is missing
    if include_corp_gov and not st.session_state.company_symbol:
        st.warning("⚠️ NSE Symbol/Ticker is required for Corporate Governance Analysis. Please go back to Step 1 to add it.")

    # Placeholder for YouTube analysis option (can be enabled later)
    # include_youtube = st.checkbox("Include Earnings Call Transcript Analysis (YouTube)",
    #                               value=st.session_state.include_youtube_transcripts,
    #                               key="cb_youtube_step2",
    #                               disabled=True) # Keep disabled until feature is ready
    # if include_youtube: st.info("YouTube analysis feature is under development.")

    st.markdown("---") # Visual separator

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Company Info"):
            logger.debug("Navigating back to Step 1 from Step 2.")
            st.session_state.step = 1
            st.rerun()
    with col2:
        # Disable 'Continue' if Corp Gov is selected without a symbol
        disable_continue = include_corp_gov and not st.session_state.company_symbol
        button_label = "Continue to Document Upload"
        if disable_continue: button_label += " (Symbol Required for Corp Gov)"

        if st.button(button_label, disabled=disable_continue):
            # Update session state with checkbox values
            st.session_state.include_corporate_governance = st.session_state.cb_governance_step2
            # st.session_state.include_youtube_transcripts = st.session_state.cb_youtube_step2 # Uncomment when ready
            logger.info(f"Analysis options selected: Corp Gov={st.session_state.include_corporate_governance}") # Log choices
            # Proceed to next step
            st.session_state.step = 3
            st.rerun()

# == Step 3: Document Analysis ==
elif st.session_state.step == 3:
    st.header("Step 3: Document Analysis (Optional)")
    logger.debug("Displaying Step 3: Document Analysis")

    # File uploader for PDF documents
    uploaded_file = st.file_uploader("Upload Company PDF Document (e.g., Annual Report, Prospectus)",
                                     type="pdf",
                                     key="pdf_uploader_step3")

    temp_dir = tempfile.gettempdir() # Use system's temporary directory

    # Process uploaded file
    if uploaded_file is not None:
        # Check if this is a new file or the same as before
        if st.session_state.uploaded_pdf != uploaded_file.name:
            logger.info(f"New file uploaded: {uploaded_file.name}, size: {uploaded_file.size} bytes")
            try:
                # Define temporary path (consider adding timestamp or unique ID for safety)
                safe_filename = re.sub(r'[\\/*?:"<>|]', "_", uploaded_file.name) # Basic sanitization
                temp_pdf_path = os.path.join(temp_dir, f"streamlit_temp_{safe_filename}")

                # Save the uploaded file buffer to the temporary path
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                logger.info(f"Saved uploaded file temporarily to: {temp_pdf_path}")

                # Update session state with new file details
                st.session_state.uploaded_pdf = uploaded_file.name
                st.session_state.pdf_path = temp_pdf_path
                st.session_state.pdf_queries = [] # Reset queries when a new file is uploaded
                st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

            except Exception as e:
                logger.error(f"Error saving uploaded file '{uploaded_file.name}': {e}", exc_info=True)
                st.error(f"❌ Error saving uploaded file: {e}")
                # Reset session state related to file upload on error
                st.session_state.uploaded_pdf = None
                st.session_state.pdf_path = None
                st.session_state.pdf_queries = []
        else:
            # File is the same as before, no need to re-save
            # logger.debug(f"File '{uploaded_file.name}' is already uploaded.")
             st.success(f"✅ Using previously uploaded file: '{st.session_state.uploaded_pdf}'")

    # Check if the temporary file still exists (it might be cleared by the OS)
    elif st.session_state.pdf_path and not os.path.exists(st.session_state.pdf_path):
         logger.warning(f"Previously uploaded PDF not found at path: {st.session_state.pdf_path}. Clearing state.")
         st.warning("⚠️ Previously uploaded file seems to be missing from temporary storage. Please upload it again if needed.")
         st.session_state.uploaded_pdf = None
         st.session_state.pdf_path = None
         st.session_state.pdf_queries = []

    # Display query input only if a PDF is successfully loaded and path exists
    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        st.subheader("Custom Queries for Document Analysis")
        st.write("Add specific questions (one per line) to ask about the content of the uploaded document.")

        # Text area for entering queries
        pdf_queries_text = st.text_area("Enter each query on a new line:",
                                     value="\n".join(st.session_state.pdf_queries),
                                     height=150,
                                     key="pdf_query_input_step3",
                                     help="These queries will be used for RAG analysis if configured.")

        # Update query list in session state based on text area content
        # This happens on rerun or when the 'Continue' button is pressed
        current_queries = [q.strip() for q in pdf_queries_text.split("\n") if q.strip()]
        if current_queries != st.session_state.pdf_queries:
             logger.debug(f"PDF queries updated: {current_queries}")
             st.session_state.pdf_queries = current_queries

    else:
         st.info("ℹ️ Upload a PDF document above to enable custom question answering about its content.")

    st.markdown("---")

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Analysis Options"):
            logger.debug("Navigating back to Step 2 from Step 3.")
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Continue to Review"):
            # Final update of queries before proceeding
            if 'pdf_query_input_step3' in st.session_state:
                 final_queries = [q.strip() for q in st.session_state.pdf_query_input_step3.split("\n") if q.strip()]
                 st.session_state.pdf_queries = final_queries
                 logger.info(f"Final PDF queries for review: {final_queries}")

            logger.debug("Proceeding to Step 4: Review.")
            st.session_state.step = 4
            st.rerun()

# == Step 4: Review Configuration ==
elif st.session_state.step == 4:
    st.header("Step 4: Review Analysis Configuration")
    logger.debug("Displaying Step 4: Review Configuration")

    # --- Build Configuration Summary for Display ---
    config_summary = {}
    config_summary["**Company Details**"] = {
        "Name": st.session_state.company_name or "_Not Provided_",
        "Industry": st.session_state.industry or "_Not Provided_",
        "NSE Symbol": st.session_state.company_symbol or "_Not Provided_",
        "Website": st.session_state.company_website or "_Not Provided_",
    }

    # Determine analysis types included
    analysis_types_list = ["News Forensic Analysis"] # Base analysis
    if st.session_state.include_corporate_governance:
        analysis_types_list.append("Corporate Governance Analysis")
    # if st.session_state.include_youtube_transcripts: # Add when ready
    #     analysis_types_list.append("Earnings Call Transcript Analysis")
    config_summary["**Analysis Scope**"] = {"Types Included": analysis_types_list}

    # Document analysis details
    doc_analysis_summary = {"Status": "_No document uploaded_"}
    if st.session_state.uploaded_pdf and st.session_state.pdf_path:
        # Check again if file exists, as review step might happen after temp cleanup
        if os.path.exists(st.session_state.pdf_path):
            doc_analysis_summary["Status"] = f"Using file: `{st.session_state.uploaded_pdf}`"
            if st.session_state.pdf_queries:
                doc_analysis_summary["Custom Queries"] = st.session_state.pdf_queries
            else:
                 doc_analysis_summary["Custom Queries"] = ["_None specified_"] # Explicitly show none
        else:
             doc_analysis_summary["Status"] = f"⚠️ Error: File `{st.session_state.uploaded_pdf}` no longer found. Please re-upload."
             logger.warning(f"File specified in state ({st.session_state.pdf_path}) not found during review step.")
             # Clear potentially invalid state
             st.session_state.pdf_path = None
             st.session_state.uploaded_pdf = None


    config_summary["**Document Analysis**"] = doc_analysis_summary

    # --- Display Configuration Sections ---
    st.subheader("Please review your analysis setup:")
    for section_title, section_details in config_summary.items():
        st.markdown(f"### {section_title}")
        if isinstance(section_details, dict):
            for key, value in section_details.items():
                # Handle lists (like analysis types or queries)
                if isinstance(value, list):
                     st.write(f"**{key}**:")
                     if not value or value == ["_None specified_"]:
                         st.write("_None_")
                     else:
                         for i, item in enumerate(value):
                             # Display queries with numbering
                             prefix = f"{i+1}. " if key == "Custom Queries" else "- "
                             st.write(f"{prefix}{item}")
                # Display simple key-value pairs
                else:
                    st.write(f"**{key}**: {value}")
        st.markdown("---") # Separator between sections

    # Store the generated summary in session state (optional, could be useful for report header)
    st.session_state.configuration = config_summary
    logger.info(f"Configuration reviewed: {config_summary}")

    # --- Validation Checks Before Starting Analysis ---
    validation_errors = []
    if not st.session_state.company_name:
        validation_errors.append("Company Name is missing (Go back to Step 1).")
    if st.session_state.include_corporate_governance and not st.session_state.company_symbol:
        validation_errors.append("NSE Symbol is required for Corporate Governance Analysis (Go back to Step 1).")
    # Add check if document vanished but was intended for use
    if st.session_state.uploaded_pdf and not st.session_state.pdf_path:
         validation_errors.append("Uploaded document file is missing. Please re-upload in Step 3.")


    # Display validation errors if any
    if validation_errors:
        st.error("🚨 Please correct the following issues before starting the analysis:")
        for error in validation_errors:
            st.write(f"- {error}")
        logger.warning(f"Validation failed before starting analysis: {validation_errors}")

    # --- Navigation Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Document Upload"):
            logger.debug("Navigating back to Step 3 from Step 4.")
            st.session_state.step = 3
            st.rerun()
    with col2:
        # Start Analysis button - disabled if validation errors exist
        start_button_label = "🚀 Start Analysis"
        if validation_errors: start_button_label = "Fix Errors to Start Analysis"
        start_button = st.button(start_button_label, disabled=bool(validation_errors), type="primary")

        if start_button:
            logger.info("'Start Analysis' button clicked.")
            # Clear previous report and reset messages for the new run
            st.session_state.report = None
            st.session_state.messages = [
                 {"role": "assistant", "content": "Analysis configuration confirmed. Starting process..."}
            ]
            logger.info("Cleared previous report and messages. Triggering analysis function.")
            # Call the main analysis function
            run_unified_analysis() # This function now handles logging and errors internally
            # run_unified_analysis sets step to 5 and finishes; Streamlit proceeds naturally.
            # No explicit rerun needed here as run_unified_analysis completes synchronously
            # However, if run_unified_analysis were async, a mechanism to rerun upon completion would be needed.
            st.rerun() # Rerun to ensure UI updates after synchronous analysis completes and state changes


# == Step 5: Results ==
elif st.session_state.step == 5:
    st.header("Analysis Results")
    logger.debug("Displaying Step 5: Results")

    # Brief check if analysis is somehow still marked running (e.g., race condition)
    if st.session_state.analysis_running:
        logger.warning("Entered Step 5, but 'analysis_running' is still True. Forcing UI update.")
        with st.spinner("Finalizing analysis results..."):
            time.sleep(1) # Small delay just in case
        st.session_state.analysis_running = False # Ensure it's False now
        st.rerun() # Rerun to show final state

    # --- Display Results using Tabs ---
    tab_log, tab_report = st.tabs(["📊 Analysis Log", "📄 Analysis Report"])

    # --- Tab 1: Analysis Log ---
    with tab_log:
        st.subheader("Analysis Process Log")
        log_container = st.container(height=600) # Use a container with fixed height for scrollability

        with log_container:
            if not st.session_state.messages:
                 log_container.info("Log is empty.")
            else:
                # Iterate through messages and display with appropriate styling
                for msg in st.session_state.messages:
                    role = msg.get("role", "assistant")
                    content = msg.get("content", "_No content_")
                    icon = "👤" if role == "user" else "🤖"

                    # Determine log level class for styling
                    log_class = "log-message "
                    if "❌" in content or "failed" in content.lower() or "error" in content.lower():
                        log_class += "log-error"
                    elif "⚠️" in content or "warning" in content.lower() or "skipped" in content.lower():
                        log_class += "log-warning"
                    elif "✅" in content or "success" in content.lower() or "complete" in content.lower():
                        log_class += "log-success"
                    elif "ℹ️" in content or "info" in content.lower() or "starting" in content.lower():
                         log_class += "log-info"
                    elif role == "user":
                        log_class += "log-user"
                    else:
                        log_class += "log-assistant"

                    # Display styled log message using markdown with class
                    log_container.markdown(f'<div class="{log_class}">{icon} **{role.capitalize()}:** {content}</div>', unsafe_allow_html=True)

    # --- Tab 2: Analysis Report ---
    with tab_report:
        st.subheader("Generated Report")

        if st.session_state.report:
            logger.debug("Report content found in session state. Displaying.")
            # Display the report content within the styled container
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown(st.session_state.report, unsafe_allow_html=True) # Allow HTML if needed
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Add Download Button ---
            company_name = st.session_state.company_name
            if company_name:
                try:
                    # Use configured reports directory
                    reports_dir = config_manager.get_path("reports_dir", default="reports")
                    combined_reports_dir = os.path.join(reports_dir, "combined_reports")

                    safe_company_name = re.sub(r'[\\/*?:"<>|]', "", company_name).replace(' ', '_')
                    os.makedirs(combined_reports_dir, exist_ok=True) # Ensure dir exists

                    # Find the latest generated PDF report for this company pattern
                    pdf_files = [f for f in os.listdir(combined_reports_dir) if f.startswith(safe_company_name) and f.endswith('.pdf')]

                    if pdf_files:
                        latest_pdf_filename = sorted(pdf_files, reverse=True)[0] # Get the most recent one
                        pdf_path_to_download = os.path.join(combined_reports_dir, latest_pdf_filename)

                        # Create a user-friendly download filename
                        # Extract timestamp if possible, otherwise use generic name
                        match = re.search(r'(\d{8}_\d{6})', latest_pdf_filename)
                        ts_part = f"_{match.group(1)}" if match else ""
                        download_filename = f"{safe_company_name}_AnalysisReport{ts_part}.pdf"

                        logger.info(f"Providing download link for: {pdf_path_to_download} as {download_filename}")
                        st.markdown(get_pdf_download_link(pdf_path_to_download, download_filename), unsafe_allow_html=True)
                    else:
                        logger.warning(f"No PDF report files found matching pattern '{safe_company_name}*.pdf' in {combined_reports_dir}")
                        st.info("ℹ️ PDF version of the report is not available for download.")

                except Exception as e:
                     logger.error(f"Error accessing report directory or finding PDF for download: {e}", exc_info=True)
                     st.error(f"Error preparing report download link: {e}")
            else:
                 logger.warning("Cannot generate download link: Company name is missing from session state.")

        # Handle cases where no report is generated
        elif st.session_state.analysis_running: # Should ideally not happen here
             st.warning("Analysis seems to be running - report is not ready yet.")
             logger.warning("Report tab accessed while analysis_running is still true.")
        else:
            logger.info("No report content available to display in Step 5.")
            # Check log messages for errors to provide context
            if any("❌" in msg["content"] for msg in st.session_state.messages):
                 st.warning("⚠️ Analysis completed with errors. Report might be incomplete or missing. Please check the 'Analysis Log' tab for details.")
            else:
                 st.info("ℹ️ Analysis complete. No report content generated. This might be expected if analysis yielded no findings or specific options weren't selected.")

    st.markdown("---")

    # --- Button to Start a New Analysis ---
    if st.button("✨ Start a New Analysis"):
        logger.info("'Start a New Analysis' button clicked. Resetting session state.")
        # Reset relevant session state variables to their defaults for a fresh start
        for key, value in default_values.items():
            st.session_state[key] = value # Reset all keys to their initial default values

        # Explicitly ensure step is 1 and provide initial message
        st.session_state.step = 1
        st.session_state.messages = [default_values["messages"][0]] # Reset log

        # Clean up temporary file if it exists
        if st.session_state.get("pdf_path") and os.path.exists(st.session_state.pdf_path):
             try:
                 os.remove(st.session_state.pdf_path)
                 logger.info(f"Removed temporary PDF file: {st.session_state.pdf_path}")
             except Exception as e:
                 logger.warning(f"Could not remove temporary PDF file {st.session_state.pdf_path}: {e}")
        # Clear path state even if removal failed
        st.session_state.pdf_path = None
        st.session_state.uploaded_pdf = None

        logger.info("Session state reset. Rerunning app to go back to Step 1.")
        st.rerun()

# --- Footer or Sidebar Elements (Optional) ---
# st.sidebar.markdown("---")
# st.sidebar.info("Financial Analysis System v1.0")
# logger.debug("Streamlit app execution finished for this run.")