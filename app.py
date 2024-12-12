import os
import streamlit as st
import pandas as pd
from pandasai.llm import OpenAI
from pandasai import Agent
from pandasai.ee.agents.judge_agent import JudgeAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PandasAISessionManager:
    """
    Manages PandasAI agent state across Streamlit sessions
    """
    @staticmethod
    def initialize_agent(uploaded_files):
        """
        Initialize PandasAI agent with uploaded Excel files
        """
        dfs = []
        for file in uploaded_files:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet)
                dfs.append(df)
        
        # Initialize LLM and Agent with persistent memory
        llm = OpenAI(
            api_token=st.secrets["OPENAI_API_KEY"],
            temperature=0,
            seed=26
        )
        
        judge = JudgeAgent(config={"llm": llm})
        agent = Agent(dfs, memory_size=10, config={"llm": llm}, judge=judge)
        
        return agent, dfs

def main():
    # Page configuration
    st.set_page_config(
        page_title="PandasAI Intelligent Analyzer", 
        page_icon="📊", 
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-container {
        background-color: #f0f2f6;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🤖 PandasAI Intelligent Data Analyzer")
    
    # Initialize session state for persistent agent
    if 'agent_state' not in st.session_state:
        st.session_state.agent_state = {
            'agent': None,
            'dfs': None,
            'conversation_history': [],
        }
    
    # Sidebar for file upload and session management
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_files = st.file_uploader(
            "Upload Excel Files", 
            type=['xlsx', 'xls'], 
            accept_multiple_files=True
        )
        
        # New Session Button
        if st.button("Start New Session"):
            st.session_state.agent_state = {
                'agent': None,
                'dfs': None,
                'conversation_history': [],
            }
    
    # Main content area
    if uploaded_files:
        # Initialize agent if not already initialized
        if st.session_state.agent_state['agent'] is None:
            st.session_state.agent_state['agent'], st.session_state.agent_state['dfs'] = \
                PandasAISessionManager.initialize_agent(uploaded_files)
        
        # Conversation interface
        st.header("💬 Intelligent Query Interface")
        
        # Query input
        user_query = st.text_input("Enter your data analysis query:")
        
        if user_query:
            try:
                # Agent exists and is ready
                agent = st.session_state.agent_state['agent']
                
                # Generate clarification questions
                clarification_questions = agent.clarification_questions(user_query)
                
                # Clarification questions section
                st.subheader("🔍 Clarification Questions")
                clarification_answers = {}
                
                for i, question in enumerate(clarification_questions, 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Q{i}: {question}**")
                    with col2:
                        skip = st.checkbox(f"Skip", key=f"skip_{i}")
                    
                    if not skip:
                        answer = st.text_input(f"Answer to Q{i}", key=f"answer_{i}")
                        clarification_answers[question] = answer
                
                # Process query button
                if st.button("Process Query"):
                    # Modify query with clarification context
                    modified_query = user_query
                    for question, answer in clarification_answers.items():
                        if answer:
                            modified_query += f" Context: {question} - {answer}"
                    
                    # Execute query using agent's memory
                    response = agent.chat(modified_query)
                    
                    # Display result
                    st.success("Analysis Result:")
                    st.write(response)
                    
                    # Explanation of agent's reasoning
                    st.subheader("🧠 Agent's Thought Process")
                    explanation = agent.explain()
                    st.info(explanation)
                    
                    # Update conversation history
                    st.session_state.agent_state['conversation_history'].append({
                        'query': modified_query,
                        'response': response
                    })
            
            except Exception as e:
                st.error(f"Analysis Error: {e}")
        
        # Conversation History in Sidebar
        with st.sidebar:
            if st.session_state.agent_state['conversation_history']:
                st.header("📜 Conversation History")
                for i, conv in enumerate(st.session_state.agent_state['conversation_history'], 1):
                    with st.expander(f"Query {i}"):
                        st.write("**Query:**", conv['query'])
                        st.write("**Response:**", conv['response'])
    
    else:
        st.warning("Please upload Excel files to begin intelligent data analysis.")

if __name__ == "__main__":
    main()
