import streamlit as st
import time
import os
import base64
import re
from datetime import datetime
import tempfile
from pathlib import Path

# Import for checkpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END

# Import both system types
from backend.core.news_forensic import NewsForensicSystem
from backend.core.corporate_governance_workflow import CorporateGovernanceSystem
from backend.utils.pdf_generator import convert_markdown_to_pdf

# Configure the page with dark theme
st.set_page_config(
    page_title="Financial Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Financial Analysis System! Please select a report type and enter company information to begin analysis."}
    ]
if "report" not in st.session_state:
    st.session_state.report = None
if "system" not in st.session_state:
    st.session_state.system = None
if "cg_system" not in st.session_state:
    st.session_state.cg_system = None
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False
if "report_type" not in st.session_state:
    st.session_state.report_type = "news_forensic"
if "company_symbol" not in st.session_state:
    st.session_state.company_symbol = ""
if "uploaded_pdf" not in st.session_state:
    st.session_state.uploaded_pdf = None

def display_pdf(pdf_path):
    """Display PDF in Streamlit"""
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def get_pdf_download_link(pdf_path, filename):
    """Generate a download link for PDF"""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode('utf-8')
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-btn">Download PDF Report</a>'

def run_analysis(company, industry):
    """Run the news forensic analysis"""
    st.session_state.system = NewsForensicSystem()
    st.session_state.analysis_running = True
    st.session_state.logs = []
    
    # Create a spinner for loading
    with st.spinner(f"Analyzing news for {company}. This may take a few minutes..."):
        # Add initial message
        st.session_state.messages.append({"role": "user", "content": f"Analyze {company}"})
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Starting news forensic analysis for {company}. Please wait..."
        })
        
        try:
            # Initialize and run the system in one shot
            system = st.session_state.system
            
            # Execute the analysis as a single-shot process
            result = system.run(company, industry, max_iterations=10)
            
            # Get path to markdown file
            report_filename = f"{company.replace(' ', '_')}_latest.md"
            markdown_path = os.path.join("markdowns", report_filename)
            
            if os.path.exists(markdown_path):
                with open(markdown_path, "r", encoding="utf-8") as f:
                    st.session_state.report = f.read()
                
                # Generate PDF for download
                pdf_path = os.path.join("markdowns", f"{company.replace(' ', '_')}_latest.pdf")
                convert_markdown_to_pdf(markdown_path, pdf_path)
                
                # Notify completion
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"✅ News forensic analysis complete for {company}."
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ Analysis finished but no report was generated. Please check the logs for errors."
                })
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"❌ Error during analysis: {str(e)}"
            })
            st.error(error_trace)
        
        finally:
            st.session_state.analysis_running = False

def run_corporate_governance(company, company_symbol, pdf_path):
    """Run the corporate governance analysis"""
    st.session_state.cg_system = CorporateGovernanceSystem()
    st.session_state.analysis_running = True
    st.session_state.logs = []
    
    # Create a spinner for loading
    with st.spinner(f"Analyzing corporate governance for {company}. This may take a few minutes..."):
        # Add initial message
        st.session_state.messages.append({"role": "user", "content": f"Analyze corporate governance for {company} ({company_symbol})"})
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Starting corporate governance analysis for {company}. Please wait..."
        })
        
        try:
            # Initialize and run the system in one shot
            system = st.session_state.cg_system
            
            # Execute the analysis as a single-shot process
            result = system.run(company, company_symbol, pdf_path, max_iterations=10)
            
            # Get path to markdown file
            report_filename = f"{company.replace(' ', '_')}_latest.md"
            markdown_path = os.path.join("corporate_reports", report_filename)
            
            if os.path.exists(markdown_path):
                with open(markdown_path, "r", encoding="utf-8") as f:
                    st.session_state.report = f.read()
                
                # Generate PDF for download
                pdf_path = os.path.join("corporate_reports", f"{company.replace(' ', '_')}_latest.pdf")
                convert_markdown_to_pdf(markdown_path, pdf_path)
                
                # Notify completion
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"✅ Corporate governance analysis complete for {company}."
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ Analysis finished but no report was generated. Please check the logs for errors."
                })
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"❌ Error during analysis: {str(e)}"
            })
            st.error(error_trace)
        
        finally:
            st.session_state.analysis_running = False

# Custom CSS
st.markdown("""
<style>
/* Global styles */
.main {
    background-color: #1E1E1E;
    color: #E0E0E0;
}

h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
}

/* For text elements in the app */
.stTextInput, .stTextArea label, .stSelectbox label {
    color: #FFFFFF !important;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #1E1E1E;
}

/* Download button */
.download-btn {
    display: inline-block;
    background-color: #2563EB;
    color: white !important;
    padding: 0.5rem 1rem;
    text-decoration: none;
    border-radius: 0.25rem;
    font-weight: 500;
    margin-top: 1rem;
    border: none;
}
.download-btn:hover {
    background-color: #1D4ED8;
    color: white !important;
}

/* Form button styling */
.stButton>button {
    background-color: #2563EB;
    color: white;
    font-weight: 500;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 0.25rem;
}
.stButton>button:hover {
    background-color: #1D4ED8;
}

/* For GitHub-like markdown styling */
.report-container {
    background-color: #0d1117;
    color: #c9d1d9;
    border-radius: 6px;
    padding: 24px;
    margin-bottom: 16px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.5;
    word-wrap: break-word;
}

/* GitHub-style headings */
.report-container h1 {
    padding-bottom: 0.3em;
    font-size: 2em;
    border-bottom: 1px solid #21262d;
    color: #e6edf3 !important;
    margin-bottom: 16px;
    font-weight: 600;
}

.report-container h2 {
    padding-bottom: 0.3em;
    font-size: 1.5em;
    border-bottom: 1px solid #21262d;
    color: #e6edf3 !important;
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
}

.report-container h3 {
    font-size: 1.25em;
    color: #e6edf3 !important;
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
}

.report-container h4 {
    font-size: 1em;
    color: #e6edf3 !important;
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
}

/* GitHub-style paragraphs */
.report-container p {
    margin-top: 0;
    margin-bottom: 16px;
}

/* GitHub-style lists */
.report-container ul, .report-container ol {
    padding-left: 2em;
    margin-top: 0;
    margin-bottom: 16px;
}

.report-container li {
    margin-top: 0.25em;
}

/* GitHub-style code blocks */
.report-container pre {
    padding: 16px;
    overflow: auto;
    font-size: 85%;
    line-height: 1.45;
    background-color: #161b22;
    border-radius: 6px;
    margin-top: 0;
    margin-bottom: 16px;
    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
}

.report-container code {
    padding: 0.2em 0.4em;
    margin: 0;
    font-size: 85%;
    background-color: #161b22;
    border-radius: 3px;
    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
}

/* GitHub-style tables */
.report-container table {
    display: block;
    width: 100%;
    overflow: auto;
    margin-top: 0;
    margin-bottom: 16px;
    border-spacing: 0;
    border-collapse: collapse;
}

.report-container table th {
    padding: 6px 13px;
    border: 1px solid #30363d;
    font-weight: 600;
    background-color: #161b22;
}

.report-container table td {
    padding: 6px 13px;
    border: 1px solid #30363d;
}

.report-container table tr {
    background-color: #0d1117;
    border-top: 1px solid #21262d;
}

.report-container table tr:nth-child(2n) {
    background-color: #161b22;
}

/* GitHub-style blockquotes */
.report-container blockquote {
    padding: 0 1em;
    color: #8b949e;
    border-left: 0.25em solid #30363d;
    margin: 0 0 16px 0;
}

/* GitHub-style horizontal rule */
.report-container hr {
    height: 0.25em;
    padding: 0;
    margin: 24px 0;
    background-color: #21262d;
    border: 0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.title("Financial Analysis System")

# Select report type
report_type = st.sidebar.radio(
    "Select Report Type",
    ["News Forensic Analysis", "Corporate Governance Analysis"],
    index=0 if st.session_state.report_type == "news_forensic" else 1
)

st.session_state.report_type = "news_forensic" if report_type == "News Forensic Analysis" else "corporate_governance"

st.sidebar.subheader("Enter Company Information")

# Different forms based on report type
if st.session_state.report_type == "news_forensic":
    with st.sidebar.form("news_forensic_form"):
        company_name = st.text_input("Company Name", key="nf_company_input")
        industry = st.text_input("Industry (optional)", key="nf_industry_input")
        submit_button = st.form_submit_button("Start News Forensic Analysis")
        
    if submit_button and company_name and not st.session_state.analysis_running:
        st.session_state.report = None
        run_analysis(company_name, industry)
else:
    with st.sidebar.form("corporate_governance_form"):
        company_name = st.text_input("Company Name", key="cg_company_input")
        company_symbol = st.text_input("Company Symbol", key="cg_symbol_input")
        uploaded_file = st.file_uploader("Upload Company Report PDF", type="pdf")
        submit_button = st.form_submit_button("Start Corporate Governance Analysis")
        
    if submit_button and company_name and company_symbol and uploaded_file and not st.session_state.analysis_running:
        st.session_state.report = None
        
        # Save the uploaded PDF to a temporary file
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        run_corporate_governance(company_name, company_symbol, pdf_path)

# Chat and report layout
col1, col2 = st.columns([1, 2])

# Chat interface (left column)
with col1:
    st.subheader("Conversation")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# Report display (right column)
with col2:
    st.subheader("Analysis Report")
    
    if st.session_state.report:
        # Create a container with padding for the report
        report_container = st.container()
        
        with report_container:
            # Apply GitHub-style markdown rendering
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            # Use st.markdown WITHOUT unsafe_allow_html to properly render markdown
            st.markdown(st.session_state.report)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add download option
        if st.session_state.report_type == "news_forensic":
            report_dir = "markdowns"
        else:
            report_dir = "corporate_reports"
            
        if company_name:
            pdf_path = os.path.join(report_dir, f"{company_name.replace(' ', '_')}_latest.pdf")
            if os.path.exists(pdf_path):
                report_type_str = "forensic" if st.session_state.report_type == "news_forensic" else "governance"
                download_filename = f"{company_name.replace(' ', '_')}_{report_type_str}_report.pdf"
                st.markdown(get_pdf_download_link(pdf_path, download_filename), unsafe_allow_html=True)
    else:
        st.info("No report available yet. Start an analysis to generate a report.")

# Display a simple spinner when analysis is running
if st.session_state.analysis_running:
    st.info("Analysis in progress... Please wait.")
    st.spinner()