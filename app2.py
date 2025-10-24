import streamlit as st
import pandas as pd
import json
import yaml
import io
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px

# Import services
from services.ai_service import AIService
from services.file_parser import FileParser
from utils.template_engine import TemplateEngine
from utils.document_generator import DocumentGenerator

# Page configuration
st.set_page_config(
    page_title="Agentic Docs Builder - Flora Edition",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Flora theme
st.markdown("""
<style>
    .main {
        background-color: #F8F8FF;
    }
    .stApp {
        background: linear-gradient(135deg, #E6E6FA 0%, #F8F8FF 100%);
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .status-success {
        background-color: #D4EDDA;
        border-color: #28A745;
        color: #155724;
    }
    .status-error {
        background-color: #F8D7DA;
        border-color: #DC3545;
        color: #721C24;
    }
    .status-running {
        background-color: #FFF3CD;
        border-color: #FFC107;
        color: #856404;
    }
    .status-info {
        background-color: #D1ECF1;
        border-color: #17A2B8;
        color: #0C5460;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(147, 112, 219, 0.1);
        border: 2px solid #E6E6FA;
    }
    .agent-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0E6FF 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 2px solid #9370DB;
        box-shadow: 0 4px 12px rgba(147, 112, 219, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'schema' not in st.session_state:
        st.session_state.schema = []
    if 'template' not in st.session_state:
        st.session_state.template = ""
    if 'generated_docs' not in st.session_state:
        st.session_state.generated_docs = []
    if 'agents_config' not in st.session_state:
        st.session_state.agents_config = load_default_agents()
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = []
    if 'follow_up_questions' not in st.session_state:
        st.session_state.follow_up_questions = []
    if 'output_format' not in st.session_state:
        st.session_state.output_format = 'txt'
    if 'ai_service' not in st.session_state:
        st.session_state.ai_service = None

def load_default_agents():
    """Load default agent configurations"""
    try:
        with open('agents.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            'agents': [
                {
                    'name': 'Summarizer',
                    'model': 'gemini-2.5-flash',
                    'provider': 'gemini',
                    'system_prompt': 'You are an expert summarizer. Create concise, informative summaries.',
                    'user_prompt': 'Summarize the following text:\n\n{{input}}',
                    'temperature': 0.3,
                    'max_tokens': 1000,
                    'top_p': 0.9
                },
                {
                    'name': 'Style Rewriter',
                    'model': 'gpt-4o-mini',
                    'provider': 'openai',
                    'system_prompt': 'You are a professional writer. Rewrite content to be clear, engaging, and professional.',
                    'user_prompt': 'Rewrite the following text in a professional tone:\n\n{{input}}',
                    'temperature': 0.7,
                    'max_tokens': 1500,
                    'top_p': 0.95
                },
                {
                    'name': 'JSON Converter',
                    'model': 'grok-3-mini',
                    'provider': 'grok',
                    'system_prompt': 'You are a data structuring expert. Convert text into well-structured JSON.',
                    'user_prompt': 'Convert the following text into structured JSON:\n\n{{input}}',
                    'temperature': 0.2,
                    'max_tokens': 2000,
                    'top_p': 0.9
                }
            ]
        }

init_session_state()

# Sidebar configuration
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/E6E6FA/9370DB?text=Flora+Docs", use_container_width=True)
    st.title("🌸 Flora Edition")
    
    st.divider()
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    with st.expander("Configure API Keys", expanded=False):
        gemini_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        grok_key = st.text_input("Grok API Key (xAI)", type="password", value=os.getenv("XAI_API_KEY", ""))
        
        if st.button("💾 Save API Keys"):
            if gemini_key or openai_key or grok_key:
                st.session_state.ai_service = AIService(
                    gemini_key=gemini_key,
                    openai_key=openai_key,
                    grok_key=grok_key
                )
                st.success("✅ API keys configured!")
            else:
                st.warning("⚠️ Please provide at least one API key")
    
    st.divider()
    
    # Output format selection
    st.subheader("📄 Output Format")
    st.session_state.output_format = st.selectbox(
        "Document Format",
        options=['txt', 'markdown', 'docx'],
        index=0
    )
    
    st.divider()
    
    # Statistics
    if st.session_state.dataset is not None:
        st.subheader("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Records", len(st.session_state.dataset))
        with col2:
            st.metric("Columns", len(st.session_state.schema))
        
        if st.session_state.generated_docs:
            st.metric("Documents", len(st.session_state.generated_docs))

# Main content
st.title("🌸 Agentic Docs Builder - Flora Edition")
st.markdown("*Automated document generation with AI-powered processing pipeline*")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data Ingestion",
    "📝 Template Definition",
    "📄 Generated Documents",
    "🤖 Agent Configuration",
    "🚀 Pipeline Execution"
])

# TAB 1: Data Ingestion
with tab1:
    st.header("📁 Data Ingestion")
    st.markdown("Upload your structured dataset (CSV, JSON, XLSX, ODS, TXT)")
    
    uploaded_file = st.file_uploader(
        "Choose a dataset file",
        type=['csv', 'json', 'xlsx', 'ods', 'txt'],
        help="Upload a structured dataset file"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Parsing dataset..."):
                parser = FileParser()
                dataset, schema = parser.parse_dataset_file(uploaded_file)
                st.session_state.dataset = dataset
                st.session_state.schema = schema
            
            st.markdown('<div class="status-box status-success">✅ Dataset loaded successfully!</div>', unsafe_allow_html=True)
            
            # Display dataset preview
            st.subheader("📊 Dataset Preview (First 10 Records)")
            df = pd.DataFrame(dataset[:10])
            st.dataframe(df, use_container_width=True, height=400)
            
            # Dataset statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Records", len(dataset))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Columns", len(schema))
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Data Types", len(set(type(v).__name__ for row in dataset for v in row.values())))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Column information
            with st.expander("📋 Column Details"):
                col_info = []
                for col in schema:
                    sample_values = [row.get(col) for row in dataset[:5] if col in row]
                    col_info.append({
                        'Column': col,
                        'Type': type(sample_values[0]).__name__ if sample_values else 'Unknown',
                        'Sample': str(sample_values[0])[:50] if sample_values else 'N/A'
                    })
                st.table(pd.DataFrame(col_info))
            
        except Exception as e:
            st.markdown(f'<div class="status-box status-error">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

# TAB 2: Template Definition
with tab2:
    st.header("📝 Template Definition")
    st.markdown("Define your document template using `{{column_name}}` placeholders")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Template Input")
        
        # Upload template file
        template_file = st.file_uploader(
            "Upload template file (optional)",
            type=['txt', 'md', 'docx'],
            key="template_upload"
        )
        
        if template_file is not None:
            try:
                parser = FileParser()
                template_content = parser.parse_template_file(template_file)
                st.session_state.template = template_content
                st.success("✅ Template loaded from file!")
            except Exception as e:
                st.error(f"❌ Error loading template: {str(e)}")
        
        # Template editor
        st.session_state.template = st.text_area(
            "Template Content",
            value=st.session_state.template,
            height=400,
            placeholder="Enter your template here...\n\nExample:\nName: {{name}}\nEmail: {{email}}\nDescription: {{description}}"
        )
        
        if st.session_state.schema:
            with st.expander("📋 Available Columns"):
                st.write(", ".join([f"`{{{{{col}}}}}`" for col in st.session_state.schema]))
    
    with col2:
        st.subheader("Live Preview")
        
        if st.session_state.template and st.session_state.dataset:
            try:
                engine = TemplateEngine()
                preview = engine.render(st.session_state.template, st.session_state.dataset[0])
                st.markdown("**Preview with first record:**")
                st.text_area("", value=preview, height=400, disabled=True, key="preview")
            except Exception as e:
                st.error(f"Preview error: {str(e)}")
        else:
            st.info("ℹ️ Upload data and create a template to see preview")
    
    st.divider()
    
    # Generate documents button
    if st.button("🚀 Generate Documents", type="primary", use_container_width=True):
        if not st.session_state.template:
            st.error("❌ Please create a template first")
        elif not st.session_state.dataset:
            st.error("❌ Please upload a dataset first")
        else:
            with st.spinner("🔄 Generating documents..."):
                try:
                    generator = DocumentGenerator()
                    docs = generator.generate(
                        st.session_state.template,
                        st.session_state.dataset,
                        st.session_state.output_format
                    )
                    st.session_state.generated_docs = docs
                    st.success(f"✅ Generated {len(docs)} documents!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Generation error: {str(e)}")

# TAB 3: Generated Documents
with tab3:
    st.header("📄 Generated Documents")
    
    if not st.session_state.generated_docs:
        st.info("ℹ️ No documents generated yet. Go to 'Template Definition' to create documents.")
    else:
        st.success(f"✅ {len(st.session_state.generated_docs)} documents ready")
        
        # Export all button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("📦 Export All Documents", type="primary"):
                generator = DocumentGenerator()
                zip_buffer = generator.export_all(st.session_state.generated_docs, st.session_state.output_format)
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_buffer,
                    file_name=f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
        st.divider()
        
        # Search and filter
        search = st.text_input("🔍 Search documents", placeholder="Search by filename or content...")
        
        # Display documents
        for idx, doc in enumerate(st.session_state.generated_docs):
            if search and search.lower() not in doc['filename'].lower() and search.lower() not in doc['content'].lower():
                continue
            
            with st.expander(f"📄 {doc['filename']}", expanded=False):
                edited_content = st.text_area(
                    "Content",
                    value=doc['content'],
                    height=200,
                    key=f"doc_{idx}"
                )
                st.session_state.generated_docs[idx]['content'] = edited_content
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.download_button(
                        label="⬇️ Download",
                        data=edited_content,
                        file_name=doc['filename'],
                        mime="text/plain"
                    )
                with col2:
                    if st.button("🗑️ Delete", key=f"del_{idx}"):
                        st.session_state.generated_docs.pop(idx)
                        st.rerun()

# TAB 4: Agent Configuration
with tab4:
    st.header("🤖 Agent Configuration")
    st.markdown("Configure your AI agent pipeline")
    
    # Model options
    MODEL_OPTIONS = {
        'gemini': ['gemini-2.5-flash', 'gemini-2.5-flash-lite'],
        'openai': ['gpt-5-nano', 'gpt-4o-mini', 'gpt-4.1-mini'],
        'grok': ['grok-4-fast-reasoning', 'grok-3-mini']
    }
    
    # Load/Save agents configuration
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        uploaded_agents = st.file_uploader("📂 Load agents.yaml", type=['yaml', 'yml'])
        if uploaded_agents:
            try:
                st.session_state.agents_config = yaml.safe_load(uploaded_agents)
                st.success("✅ Agents configuration loaded!")
            except Exception as e:
                st.error(f"❌ Error loading config: {str(e)}")
    
    with col2:
        if st.button("💾 Save Config"):
            yaml_str = yaml.dump(st.session_state.agents_config, default_flow_style=False)
            st.download_button(
                label="⬇️ Download YAML",
                data=yaml_str,
                file_name="agents.yaml",
                mime="text/yaml"
            )
    
    with col3:
        if st.button("➕ Add Agent"):
            new_agent = {
                'name': f'Agent {len(st.session_state.agents_config["agents"]) + 1}',
                'model': 'gemini-2.5-flash',
                'provider': 'gemini',
                'system_prompt': 'You are a helpful AI assistant.',
                'user_prompt': '{{input}}',
                'temperature': 0.7,
                'max_tokens': 1000,
                'top_p': 0.9
            }
            st.session_state.agents_config['agents'].append(new_agent)
            st.rerun()
    
    st.divider()
    
    # Agent configuration UI
    for idx, agent in enumerate(st.session_state.agents_config['agents']):
        st.markdown(f'<div class="agent-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader(f"🤖 Agent {idx + 1}: {agent['name']}")
        with col2:
            if st.button("🗑️", key=f"delete_agent_{idx}"):
                st.session_state.agents_config['agents'].pop(idx)
                st.rerun()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            agent['name'] = st.text_input(
                "Agent Name",
                value=agent['name'],
                key=f"name_{idx}"
            )
        
        with col2:
            agent['provider'] = st.selectbox(
                "Provider",
                options=['gemini', 'openai', 'grok'],
                index=['gemini', 'openai', 'grok'].index(agent['provider']),
                key=f"provider_{idx}"
            )
        
        with col3:
            agent['model'] = st.selectbox(
                "Model",
                options=MODEL_OPTIONS[agent['provider']],
                index=MODEL_OPTIONS[agent['provider']].index(agent['model']) if agent['model'] in MODEL_OPTIONS[agent['provider']] else 0,
                key=f"model_{idx}"
            )
        
        agent['system_prompt'] = st.text_area(
            "System Prompt",
            value=agent['system_prompt'],
            height=100,
            key=f"sys_{idx}"
        )
        
        agent['user_prompt'] = st.text_area(
            "User Prompt (use {{input}} for previous output)",
            value=agent['user_prompt'],
            height=100,
            key=f"user_{idx}"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            agent['temperature'] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(agent['temperature']),
                step=0.1,
                key=f"temp_{idx}"
            )
        with col2:
            agent['max_tokens'] = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=10000,
                value=int(agent['max_tokens']),
                step=100,
                key=f"tokens_{idx}"
            )
        with col3:
            agent['top_p'] = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=float(agent['top_p']),
                step=0.05,
                key=f"topp_{idx}"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()

# TAB 5: Pipeline Execution
with tab5:
    st.header("🚀 Pipeline Execution")
    
    if not st.session_state.ai_service:
        st.warning("⚠️ Please configure API keys in the sidebar first!")
    else:
        # Input section
        st.subheader("📥 Input")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 Use Doc 1") and st.session_state.generated_docs:
                st.session_state.pipeline_input = st.session_state.generated_docs[0]['content']
        with col2:
            if st.button("📄 Use Doc 2") and len(st.session_state.generated_docs) > 1:
                st.session_state.pipeline_input = st.session_state.generated_docs[1]['content']
        with col3:
            if st.button("📄 Use Doc 3") and len(st.session_state.generated_docs) > 2:
                st.session_state.pipeline_input = st.session_state.generated_docs[2]['content']
        
        pipeline_input = st.text_area(
            "Input Text",
            value=st.session_state.get('pipeline_input', ''),
            height=200,
            placeholder="Enter the text to process through the agent pipeline..."
        )
        
        # Execute button
        execute_col1, execute_col2 = st.columns([3, 1])
        with execute_col1:
            execute_btn = st.button(
                "▶️ Execute Pipeline",
                type="primary",
                use_container_width=True,
                disabled=not pipeline_input or not st.session_state.agents_config['agents']
            )
        with execute_col2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.pipeline_results = []
                st.session_state.follow_up_questions = []
                st.rerun()
        
        if execute_btn:
            st.session_state.pipeline_results = []
            st.session_state.follow_up_questions = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Pipeline visualization
            st.divider()
            st.subheader("📊 Pipeline Progress")
            
            chart_placeholder = st.empty()
            
            current_input = pipeline_input
            total_agents = len(st.session_state.agents_config['agents'])
            
            for idx, agent in enumerate(st.session_state.agents_config['agents']):
                progress = (idx + 1) / total_agents
                progress_bar.progress(progress)
                status_text.markdown(f'<div class="status-box status-running">🔄 Running: {agent["name"]} ({idx + 1}/{total_agents})</div>', unsafe_allow_html=True)
                
                try:
                    # Execute agent
                    result = st.session_state.ai_service.run_agent(agent, current_input)
                    
                    st.session_state.pipeline_results.append({
                        'agent': agent['name'],
                        'model': f"{agent['provider']}/{agent['model']}",
                        'input': current_input[:200] + '...' if len(current_input) > 200 else current_input,
                        'output': result,
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    current_input = result
                    
                    # Update visualization
                    fig = create_pipeline_visualization(st.session_state.pipeline_results, total_agents)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.session_state.pipeline_results.append({
                        'agent': agent['name'],
                        'model': f"{agent['provider']}/{agent['model']}",
                        'input': current_input[:200] + '...' if len(current_input) > 200 else current_input,
                        'output': '',
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    status_text.markdown(f'<div class="status-box status-error">❌ Error in {agent["name"]}: {str(e)}</div>', unsafe_allow_html=True)
                    break
            
            # Generate follow-up questions
            if st.session_state.pipeline_results and st.session_state.pipeline_results[-1]['status'] == 'success':
                status_text.markdown('<div class="status-box status-running">🔄 Generating follow-up questions...</div>', unsafe_allow_html=True)
                try:
                    final_output = st.session_state.pipeline_results[-1]['output']
                    questions = st.session_state.ai_service.generate_follow_up_questions(final_output)
                    st.session_state.follow_up_questions = questions
                except Exception as e:
                    st.warning(f"⚠️ Could not generate follow-up questions: {str(e)}")
                
                progress_bar.progress(1.0)
                status_text.markdown('<div class="status-box status-success">✅ Pipeline completed successfully!</div>', unsafe_allow_html=True)
                st.balloons()
        
        # Display results
        if st.session_state.pipeline_results:
            st.divider()
            st.subheader("📋 Pipeline Results")
            
            for idx, result in enumerate(st.session_state.pipeline_results):
                status_icon = "✅" if result['status'] == 'success' else "❌"
                status_class = "status-success" if result['status'] == 'success' else "status-error"
                
                with st.expander(f"{status_icon} Step {idx + 1}: {result['agent']} ({result['model']})", expanded=(idx == len(st.session_state.pipeline_results) - 1)):
                    if result['status'] == 'success':
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Input:**")
                            st.text_area("", value=result['input'], height=150, disabled=True, key=f"in_{idx}")
                        with col2:
                            st.markdown("**Output:**")
                            st.text_area("", value=result['output'], height=150, disabled=True, key=f"out_{idx}")
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
            
            # Follow-up questions
            if st.session_state.follow_up_questions:
                st.divider()
                st.subheader("💡 Follow-up Questions")
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                for i, question in enumerate(st.session_state.follow_up_questions, 1):
                    st.markdown(f"**{i}.** {question}")
                st.markdown('</div>', unsafe_allow_html=True)

def create_pipeline_visualization(results, total_agents):
    """Create an interactive pipeline visualization"""
    fig = go.Figure()
    
    # Create nodes for each agent
    x_pos = []
    y_pos = []
    colors = []
    text = []
    
    for idx, result in enumerate(results):
        x_pos.append(idx)
        y_pos.append(1)
        colors.append('#28A745' if result['status'] == 'success' else '#DC3545')
        text.append(f"{result['agent']}<br>{result['model']}")
    
    # Add remaining agents
    for idx in range(len(results), total_agents):
        x_pos.append(idx)
        y_pos.append(1)
        colors.append('#E6E6FA')
        text.append(f"Agent {idx + 1}<br>Pending")
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=50,
            color=colors,
            line=dict(color='#9370DB', width=3)
        ),
        text=text,
        textposition="bottom center",
        textfont=dict(size=10),
        hoverinfo='text'
    ))
    
    # Add connections
    for i in range(len(x_pos) - 1):
        fig.add_trace(go.Scatter(
            x=[x_pos[i], x_pos[i+1]],
            y=[y_pos[i], y_pos[i+1]],
            mode='lines',
            line=dict(color='#9370DB', width=2),
            hoverinfo='none',
            showlegend=False
        ))
    
    fig.update_layout(
        title="Agent Pipeline Flow",
        showlegend=False,
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with 💜 by Flora Edition")
    st.sidebar.markdown(f"Version 1.0.0 | {datetime.now().strftime('%Y-%m-%d')}")
