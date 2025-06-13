import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tempfile
import os
import docx
import fitz  # PyMuPDF
import pandas as pd
from langdetect import detect, LangDetectException
from gingerit import GingerIt  # Replaced language-tool-python with gingerit
from dotenv import load_dotenv
from typing import Tuple, List

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="Essay Grader", layout="wide")
st.title("📝 Automated Essay Grading System (Multilingual + Grammar Highlights)")

# Constants
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese'
}

# Sidebar configuration
st.sidebar.header("Configuration")
max_score = st.sidebar.selectbox("Maximum Score", [5, 10, 20, 50, 100], index=0)
level = st.sidebar.selectbox("Student Level", ["Beginner", "Intermediate", "Advanced"], index=0)

# Rubric definitions (unchanged)
rubrics = {
    "Beginner": {
        "title": "Beginner Level Rubric",
        "criteria": [
            "1. Grammar and Mechanics (20%)",
            "2. Structure and Organization (20%)",
            "3. Relevance to Topic (20%)",
            "4. Vocabulary Usage (20%)",
            "5. Clarity and Readability (20%)"
        ]
    },
    "Intermediate": {
        "title": "Intermediate Level Rubric",
        "criteria": [
            "1. Grammar Accuracy (20%)",
            "2. Logical Structure (20%)",
            "3. Depth of Analysis (20%)",
            "4. Vocabulary Range (20%)",
            "5. Coherence and Argumentation (20%)"
        ]
    },
    "Advanced": {
        "title": "Advanced Level Rubric",
        "criteria": [
            "1. Grammar and Syntax Precision (20%)",
            "2. Structural Sophistication (20%)",
            "3. Argument Depth and Insight (20%)",
            "4. Lexical Choice and Style (20%)",
            "5. Critical Thinking and Originality (20%)"
        ]
    }
}

def display_rubric(level: str) -> str:
    """Display the rubric for the selected level"""
    rubric = rubrics[level]
    st.sidebar.subheader(rubric["title"])
    st.sidebar.write("\n".join(rubric["criteria"]))
    return "\n".join(rubric["criteria"])

rubric = display_rubric(level)

# Initialize LLM (unchanged)
def initialize_llm() -> HuggingFaceHub:
    """Initialize the language model with error handling"""
    try:
        return HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_kwargs={
                "temperature": 0.5,
                "max_new_tokens": 512,
                "do_sample": True
            }
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        st.stop()

llm = initialize_llm()

# Prompt template (unchanged)
prompt = PromptTemplate(
    input_variables=["essay", "criteria", "max_score", "language"],
    template="""As an expert essay evaluator fluent in {language}, please evaluate this essay according to the specified rubric. 

**Essay Language**: {language}
**Maximum Possible Score**: {max_score}
**Evaluation Rubric**:
{criteria}

**Essay Content**:
{essay}

**Evaluation Requirements**:
1. Final Score: [Score]/{max_score} (must be a number between 0 and {max_score})
2. Detailed Feedback:
   - Strengths identified (bullet points)
   - Areas needing improvement (with specific examples)
   - Grammar/syntax issues found
   - Suggestions for enhancement
3. Breakdown of scores per rubric category
4. Estimated grade level (e.g., A, B, C) if applicable

Please provide your evaluation in clear, structured markdown format."""
)

chain = LLMChain(prompt=prompt, llm=llm)

# File handlers (unchanged)
def extract_text_from_pdf(file) -> str:
    """Extract text content from PDF file"""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file) -> str:
    """Extract text content from DOCX file"""
    try:
        doc = docx.Document(file)
        return " ".join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

# Updated grammar checker using gingerit
def highlight_grammar_issues(text: str) -> Tuple[str, List]:
    """Highlight grammar issues in the text using gingerit"""
    if not text.strip():
        return "", []
    
    try:
        parser = GingerIt()
        result = parser.parse(text)
        highlighted = text
        corrections = []
        
        # Process each correction
        for mistake in result.get('corrections', []):
            start = mistake['start']
            end = mistake['end']
            suggestion = mistake.get('text', 'No suggestion')
            
            # Highlight the mistake
            highlighted = (
                highlighted[:start] + 
                f"<span style='background-color: #ffcccc; border-bottom: 1px dashed red;' title='Suggested: {suggestion}'>{highlighted[start:end]}</span>" + 
                highlighted[end:]
            )
            corrections.append({
                'text': highlighted[start:end],
                'suggestion': suggestion,
                'start': start,
                'end': end
            })
        
        return highlighted, corrections
    except Exception as e:
        st.error(f"Grammar check failed: {str(e)}")
        return text, []
        
# Essay input section with better UX
def get_essay_input() -> str:
    """Get essay input from either file upload or text area"""
    st.subheader("📄 Upload Essay (PDF/DOCX) or Paste Text")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"], 
                                   help="Supported formats: PDF and DOCX")
    essay_input = ""
    
    if uploaded_file:
        with st.spinner("Extracting text from file..."):
            if uploaded_file.name.endswith(".pdf"):
                essay_input = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".docx"):
                essay_input = extract_text_from_docx(uploaded_file)
    
    essay_input_manual = st.text_area("Or paste your essay here:", height=300,
                                    help="Type or paste your essay directly")
    if essay_input_manual.strip():
        essay_input = essay_input_manual.strip()
    
    return essay_input

essay_input = get_essay_input()

# Initialize session state for grades storage
if "grades" not in st.session_state:
    st.session_state.grades = []

# Enhanced evaluation function
def evaluate_essay() -> None:
    """Evaluate the essay and store results"""
    if not essay_input.strip():
        st.warning("Please upload or paste an essay first.")
        return
    
    # Detect language
    with st.spinner("Detecting language..."):
        try:
            detected_lang = detect(essay_input)
            if detected_lang in SUPPORTED_LANGUAGES:
                lang_name = SUPPORTED_LANGUAGES[detected_lang]
                st.success(f"Detected language: {lang_name}")
            else:
                detected_lang = "en"
                st.warning("Language not fully supported, defaulting to English")
        except LangDetectException:
            detected_lang = "en"
            st.warning("Could not detect language, defaulting to English")
    
    # Evaluate essay
    with st.spinner("Evaluating essay..."):
        try:
            result = chain.run(
                essay=essay_input,
                criteria=rubric,
                max_score=max_score,
                language=detected_lang
            )
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.stop()
    
    # Grammar check
    with st.spinner("Checking grammar..."):
        highlighted_text, grammar_matches = highlight_grammar_issues(essay_input)
    
    # Display results
    st.success("✅ Evaluation complete!")
    
    # Evaluation summary
    st.subheader("📊 Evaluation Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Detected Language**: `{SUPPORTED_LANGUAGES.get(detected_lang, 'English')}`")
    with col2:
        st.markdown(f"**Grammar Issues Found**: `{len(grammar_matches)}`")
    
    # Grammar highlights
    st.subheader("✨ Grammar Issues Highlighted")
    st.markdown(highlighted_text, unsafe_allow_html=True)
    
    # Feedback
    st.subheader("🗣️ Detailed Feedback")
    st.markdown(result)
    
    # Extract score
    score = "N/A"
    for line in result.split("\n"):
        if "Final Score" in line or "Score" in line:
            score = line.split(":")[-1].strip()
            break
    
    # Store results
    st.session_state.grades.append({
        "Level": level,
        "Language": SUPPORTED_LANGUAGES.get(detected_lang, "English"),
        "Max Score": max_score,
        "Score": score,
        "Grammar Issues": len(grammar_matches),
        "Feedback": result
    })

# Admin Panel functions
def export_grades() -> None:
    """Export grades to CSV"""
    if st.session_state.grades:
        df = pd.DataFrame(st.session_state.grades)
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="essay_grades.csv",
            mime="text/csv",
            help="Download all evaluation results as CSV"
        )
    else:
        st.sidebar.warning("No grades to export yet.")

def clear_grades() -> None:
    """Clear all stored grades"""
    st.session_state.grades = []
    st.sidebar.success("Grades cleared!")

# Main execution flow
if st.button("Evaluate Essay", help="Start the evaluation process"):
    evaluate_essay()

# Admin Panel
st.sidebar.header("📁 Admin Panel")
st.sidebar.button("Export Grades", on_click=export_grades)
st.sidebar.button("Clear Grades", on_click=clear_grades)
