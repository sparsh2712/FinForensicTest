import streamlit as st
import time
import os
import base64
import re
import logging
from datetime import datetime
import tempfile
from pathlib import Path

# Import for checkpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END

# Import your existing system
from backend.core.news_forensic import NewsForensicSystem
# from backend.utils.pdf_generator import convert_markdown_to_pdf

# Configure the page
st.set_page_config(
    page_title="FinForensic System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the FinForensic System! Please enter a company name to begin analysis."}
    ]
if "report" not in st.session_state:
    st.session_state.report = None
if "system" not in st.session_state:
    st.session_state.system = None
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "current_iteration" not in st.session_state:
    st.session_state.current_iteration = 0
if "max_iterations" not in st.session_state:
    st.session_state.max_iterations = 20
if "current_goto" not in st.session_state:
    st.session_state.current_goto = ""
if "logs" not in st.session_state:
    st.session_state.logs = []
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = {
        "Meta Agent": "",
        "Research Agent": "",
        "Analyst Agent": ""
    }

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

def update_log(message):
    """Add a log message and update the display"""
    if "logs" in st.session_state:
        st.session_state.logs.append(message)
        
        # Parse the message to update agent status
        patterns = [
            (r"\[Meta Agent\] (.*)", "Meta Agent"),
            (r"\[Research Agent\] (.*)", "Research Agent"),
            (r"\[Analyst Agent\] (.*)", "Analyst Agent")
        ]
        
        for pattern, agent_name in patterns:
            match = re.search(pattern, message)
            if match:
                agent_message = match.group(1)
                st.session_state.active_agent = agent_name
                st.session_state.agent_messages[agent_name] = agent_message
                break

def display_agent_status():
    """Display the current status of each agent"""
    cols = st.columns(3)
    
    agents = ["Meta Agent", "Research Agent", "Analyst Agent"]
    
    for i, agent_name in enumerate(agents):
        with cols[i]:
            is_active = st.session_state.active_agent == agent_name
            status_color = "#4CAF50" if is_active else "#9E9E9E"
            status_text = "Active" if is_active else "Waiting"
            message = st.session_state.agent_messages.get(agent_name, "")
            
            st.markdown(f"""
            <div class="agent-card" style="border-top: 4px solid {status_color};">
                <h3>{agent_name}</h3>
                <div class="agent-status" style="color: {status_color};">
                    {status_text}
                </div>
                <div class="agent-message">
                    {message}
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_logs():
    """Display the logs with styling based on agent"""
    if not st.session_state.logs:
        return
    
    with st.expander("Analysis Logs", expanded=True):
        for log in st.session_state.logs[-15:]:  # Show last 15 logs
            if "[Meta Agent]" in log:
                st.markdown(f'<div class="log-entry meta-agent">{log}</div>', unsafe_allow_html=True)
            elif "[Research Agent]" in log:
                st.markdown(f'<div class="log-entry research-agent">{log}</div>', unsafe_allow_html=True)
            elif "[Analyst Agent]" in log:
                st.markdown(f'<div class="log-entry analyst-agent">{log}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="log-entry">{log}</div>', unsafe_allow_html=True)

def run_analysis(company, industry):
    """Run the analysis directly (no threading)"""
    st.session_state.system = NewsForensicSystem()
    st.session_state.analysis_running = True
    st.session_state.progress = 0
    st.session_state.logs = []
    st.session_state.active_agent = None
    st.session_state.agent_messages = {
        "Meta Agent": "",
        "Research Agent": "",
        "Analyst Agent": ""
    }
    
    # Set up a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create containers for dynamic content
    agent_status_container = st.empty()
    logs_container = st.empty()
    
    # Add initial message
    st.session_state.messages.append({"role": "user", "content": f"Analyze {company}"})
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"Starting analysis for {company}. This may take a few minutes..."
    })
    
    try:
        # Create a custom log handler to capture logs
        class StreamHandler(logging.StreamHandler):
            def emit(self, record):
                log_entry = self.format(record)
                update_log(log_entry)
                
        # Add our custom handler to the logger
        logger = logging.getLogger("news_forensic")
        handler = StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        
        # Initialize the system
        system = st.session_state.system
        graph = system.build_graph()
        
        initial_state = {
            "company": company,
            "industry": industry,
            "research_results": {},
            "analysis_results": {},
            "analyst_status": "",
            "final_report": "",
            "report_sections": [],
            "start_time": datetime.now().isoformat(),
            "iterations": 0
        }
        
        # Create the checkpoint saver
        checkpoint_saver = MemorySaver()
        system.app = system.graph.compile(checkpointer=checkpoint_saver)
        
        system._save_state_snapshot(initial_state, "initial")
        
        update_log(f"[System] Starting analysis for {company}")
        
        current_state = system.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"{company}_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
        )
        
        iteration = 0
        max_iterations = 20
        
        while current_state.get("goto") != END and iteration < max_iterations:
            iteration += 1
            progress = min(iteration / max_iterations, 1.0)
            
            # Update progress
            progress_bar.progress(progress)
            st.session_state.progress = progress
            st.session_state.current_iteration = iteration
            st.session_state.max_iterations = max_iterations
            st.session_state.current_goto = current_state.get("goto", "processing")
            
            # Update status text
            status_text.text(f"Iteration {iteration}/{max_iterations}: {st.session_state.current_goto}")
            
            # Update agent status display
            with agent_status_container:
                display_agent_status()
            
            # Update logs display
            with logs_container:
                display_logs()
            
            # Add important updates to chat
            # Check for significant logs to update the chat
            recent_logs = st.session_state.logs[-5:] if st.session_state.logs else []
            significant_updates = [
                log for log in recent_logs if any(
                    marker in log for marker in [
                        "quality score", 
                        "Executing search query", 
                        "Generating report", 
                        "Analysis complete"
                    ]
                )
            ]
            
            if significant_updates:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": significant_updates[-1]
                })
            elif iteration % 5 == 0:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Analysis in progress ({int(progress*100)}% complete)..."
                })
            
            current_state["iterations"] = iteration
            
            # Trigger a rerun to refresh UI
            time.sleep(0.1)
            st.experimental_rerun()
            
            current_state = system.app.invoke(
                current_state,
                config={"configurable": {"thread_id": f"{company}_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
            )
        
        if iteration >= max_iterations and current_state.get("goto") != END:
            current_state["warning"] = f"Analysis terminated after reaching maximum iterations ({max_iterations})"
            current_state["goto"] = END
        
        system._save_state_snapshot(current_state, "final")
        system._save_final_report(current_state)
        system.final_state = current_state
        
        # Get path to markdown file
        report_filename = f"{company.replace(' ', '_')}_latest.md"
        markdown_path = os.path.join("markdowns", report_filename)
        
        if os.path.exists(markdown_path):
            with open(markdown_path, "r", encoding="utf-8") as f:
                st.session_state.report = f.read()
            
            # Generate PDF for download
            pdf_path = os.path.join("markdowns", f"{company.replace(' ', '_')}_latest.pdf")
            # convert_markdown_to_pdf(markdown_path, pdf_path)
            
            # Notify completion
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"✅ Analysis complete! The report for {company} is now available."
            })
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
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
        # Remove our custom handler
        for handler in logger.handlers[:]:
            if isinstance(handler, StreamHandler):
                logger.removeHandler(handler)

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1E3A8A;
    margin-bottom: 1rem;
}
.report-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1E3A8A;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #E5E7EB;
}
.download-btn {
    display: inline-block;
    background-color: #2563EB;
    color: white;
    padding: 0.5rem 1rem;
    text-decoration: none;
    border-radius: 0.25rem;
    font-weight: 500;
    margin-top: 1rem;
}
.download-btn:hover {
    background-color: #1D4ED8;
}
.stButton>button {
    background-color: #2563EB;
    color: white;
    font-weight: 500;
}
.stButton>button:hover {
    background-color: #1D4ED8;
}

/* Agent status cards */
.agent-card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 16px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}
.agent-card:hover {
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}
.agent-card h3 {
    margin-top: 0;
    font-size: 1.2rem;
    color: #1E3A8A;
}
.agent-status {
    font-weight: bold;
    margin-bottom: 8px;
}
.agent-message {
    font-size: 0.9rem;
    line-height: 1.4;
    color: #4B5563;
    height: 80px;
    overflow-y: auto;
}

/* Log entries styling */
.log-entry {
    padding: 8px;
    margin: 4px 0;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.85rem;
    background-color: #F3F4F6;
}
.log-entry.meta-agent {
    border-left: 4px solid #3B82F6;
    background-color: #EFF6FF;
}
.log-entry.research-agent {
    border-left: 4px solid #10B981;
    background-color: #ECFDF5;
}
.log-entry.analyst-agent {
    border-left: 4px solid #F59E0B;
    background-color: #FFFBEB;
}

/* Custom scrollbar for logs */
.log-container {
    max-height: 200px;
    overflow-y: auto;
    border-radius: 6px;
    border: 1px solid #E5E7EB;
}
.log-container::-webkit-scrollbar {
    width: 8px;
}
.log-container::-webkit-scrollbar-track {
    background: #F9FAFB;
}
.log-container::-webkit-scrollbar-thumb {
    background-color: #D1D5DB;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.title("FinForensic System")
st.sidebar.subheader("Enter Company Information")

with st.sidebar.form("company_form"):
    company_name = st.text_input("Company Name", key="company_input")
    industry = st.text_input("Industry (optional)", key="industry_input")
    submit_button = st.form_submit_button("Start Analysis")

if submit_button and company_name and not st.session_state.analysis_running:
    st.session_state.report = None
    run_analysis(company_name, industry)

# Main page
st.markdown('<h1 class="main-title">Financial Forensic Analysis</h1>', unsafe_allow_html=True)

# Display progress information if analysis is running
if st.session_state.analysis_running:
    st.subheader("Analysis Status")
    
    # Progress information
    progress_text = f"Iteration {st.session_state.current_iteration}/{st.session_state.max_iterations}"
    if st.session_state.current_goto:
        progress_text += f": {st.session_state.current_goto}"
    st.text(progress_text)
    
    # Progress bar
    st.progress(st.session_state.progress)
    
    # Agent status
    display_agent_status()
    
    # Display logs
    display_logs()

# Chat interface and report (side by side)
chat_col, report_col = st.columns([1, 2])

with chat_col:
    st.subheader("Conversation")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# Report display (right column)
with report_col:
    st.markdown('<h2 class="report-title">Analysis Report</h2>', unsafe_allow_html=True)
    
    if st.session_state.report:
        tab1, tab2 = st.tabs(["Markdown", "PDF"])
        
        with tab1:
            st.markdown(st.session_state.report)
        
        with tab2:
            # Convert markdown to PDF for viewing
            with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_md:
                temp_md_path = temp_md.name
                temp_md.write(st.session_state.report.encode('utf-8'))
            
            temp_pdf_path = temp_md_path.replace('.md', '.pdf')
            # convert_markdown_to_pdf(temp_md_path, temp_pdf_path)
            
            # Display PDF
            # display_pdf(temp_pdf_path)
            
            # Download link
            company = company_name if company_name else "report"
            download_filename = f"{company.replace(' ', '_')}_report.pdf"
            # st.markdown(get_pdf_download_link(temp_pdf_path, download_filename), unsafe_allow_html=True)
            
            # Clean up temp files
            try:
                os.unlink(temp_md_path)
            except:
                pass
    else:
        st.info("No report available yet. Start an analysis to generate a report.")

# Footer
st.markdown("---")
st.caption("FinForensic System © 2025")