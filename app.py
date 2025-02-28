import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from analyzer import extract_text_from_pdf, RAGResumeAnalyzer

# Set page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .upload-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-box-positive {
        background-color: #E8F5E9;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
    }
    .status-box-negative {
        background-color: #FFEBEE;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F44336;
    }
    .candidate-name {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1565C0;
    }
    .contact-info {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .assessment-box {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 5px solid #1976D2;
    }
    .link-box {
        background-color: #FAFAFA;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    .experience-badge-success {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 1rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .experience-badge-failure {
        background-color: #F44336;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 1rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main page
    st.markdown("<h1 class='main-header'>AI Resume Analyzer</h1>", unsafe_allow_html=True)
    
    # Centered upload section
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if uploaded_file:
        # Button to analyze resume
        analyze_button = st.button("Analyze Resume", type="primary", use_container_width=True)
        
        if analyze_button:
            # Show processing message
            with st.spinner("Analyzing resume... This may take a minute."):
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Extract text from PDF
                    resume_text = extract_text_from_pdf(tmp_file_path)
                    
                    # Initialize analyzer
                    analyzer = RAGResumeAnalyzer()
                    
                    # Setup RAG system
                    analyzer.setup_rag_system(resume_text)
                    
                    # Analyze resume
                    result = analyzer.analyze_resume()
                    
                    # Store results in session state
                    st.session_state.result = result
                    st.session_state.resume_analyzed = True
                    st.session_state.analyzer = analyzer
                    st.session_state.conversation_chain = analyzer.conversation_chain
                
                except Exception as e:
                    st.error(f"Error analyzing resume: {str(e)}")
                
                finally:
                    # Clean up the temp file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                
                # Rerun to update UI
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display results if available
    if hasattr(st.session_state, 'resume_analyzed') and st.session_state.resume_analyzed:
        result = st.session_state.result
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # Create columns for layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("<h2 class='sub-header'>Candidate Details</h2>", unsafe_allow_html=True)
                
                candidate_details = result['candidate_details']
                st.markdown(f"<div class='info-box'><p class='candidate-name'>{candidate_details['name']}</p>", unsafe_allow_html=True)
                
                # Show contact information
                st.markdown("<p class='contact-info'><b>📞 Phone:</b> {}</p>".format(candidate_details['phone']), unsafe_allow_html=True)
                st.markdown("<p class='contact-info'><b>📧 Email:</b> {}</p>".format(candidate_details['email']), unsafe_allow_html=True)
                st.markdown("<p class='contact-info'><b>📍 Address:</b> {}</p>".format(candidate_details['address']), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show links if available
                if candidate_details['links']:
                    st.markdown("<div class='link-box'>", unsafe_allow_html=True)
                    st.markdown("<b>🔗 Professional Links:</b>", unsafe_allow_html=True)
                    for link in candidate_details['links']:
                        if link.strip() and link.lower() != "not found":
                            st.markdown(f"- {link}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Experience and recommendation
                st.markdown("<h3>Experience & Match</h3>", unsafe_allow_html=True)
                
                # Add experience with min requirement badge
                meets_exp = result['meets_experience']
                experience_badge = "experience-badge-success" if meets_exp else "experience-badge-failure"
                req_text = "Meets requirement" if meets_exp else "Below requirement"
                
                st.markdown(
                    f"<p><b>Years of Experience:</b> {result['years_experience']:.1f} <span class='{experience_badge}'>{req_text}</span></p>",
                    unsafe_allow_html=True
                )
                
                st.markdown(f"<p><b>Matching Skill Categories:</b> {result['matching_categories']}/12</p>", unsafe_allow_html=True)
                
                # Recommendation status
                if result['recommended']:
                    st.markdown("<div class='status-box-positive'><b>✅ Recommendation:</b> Proceed with interview</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='status-box-negative'><b>⚠️ Recommendation:</b> Does not meet minimum requirements</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<h2 class='sub-header'>HR Assessment</h2>", unsafe_allow_html=True)
                
                # Display HR assessment
                st.markdown(f"<div class='assessment-box'>{result['hr_assessment']}</div>", unsafe_allow_html=True)
            
            # Q&A Section
            st.markdown("<h2 class='sub-header'>Ask about this resume</h2>", unsafe_allow_html=True)
            
            # Initialize chat history if not exists
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Get user question
            user_question = st.text_input("Ask a specific question about this resume:")
            
            if st.button("Submit Question", use_container_width=True):
                if not user_question:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Processing..."):
                        try:
                            # Get answer from RAG model
                            conversation_chain = st.session_state.conversation_chain
                            response = conversation_chain({"question": user_question})
                            
                            # Add to chat history
                            st.session_state.chat_history.append({"question": user_question, "answer": response['answer']})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("<h3>Conversation History</h3>", unsafe_allow_html=True)
                for i, exchange in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q: {exchange['question']}", expanded=(i == len(st.session_state.chat_history) - 1)):
                        st.markdown(exchange['answer'])

if __name__ == "__main__":
    main()