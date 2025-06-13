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
from textblob import TextBlob
from dotenv import load_dotenv
from typing import Tuple, List
import re  # for score extraction

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

# Rubric definitions
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
    rubric = rubrics[level]
    st.sidebar.subheader(rubric["title"])
    st.sidebar.write("\n".join(rubric["criteria"]))
    return "\n".join(rubric["criteria"])

rubric = display_rubric(level)

# Initialize LLM
def initialize_llm() -> HuggingFaceHub:
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

# Prompt template (updated)
prompt = PromptTemplate(
    input_variables=["essay", "criteria", "max_score", "language"],
    template="""
As an expert essay evaluator fluent in {language}, please evaluate this essay according to the rubric.

Return your response **strictly in this markdown structure**:
---
**Final Score**: X/{max_score}
**Grade**: A/B/C/...

**Breakdown**:
- Grammar and Mechanics: X/X
- Structure and Organization: X/X
- Relevance to Topic: X/X
- Vocabulary Usage: X/X
- Clarity and Readability: X/X

**Strengths**:
- Bullet 1
- Bullet 2

**Areas for Improvement**:
- Bullet 1
- Bullet 2

**Grammar/Syntax Issues**:
- List of issues if any

**Suggestions for Improvement**:
- Bullet 1
- Bullet 2

---
**Essay Language**: {language}
**Maximum Possible Score**: {max_score}
**Evaluation Rubric**:
{criteria}

**Essay Content**:
{essay}
"""
)

chain = LLMChain(prompt=prompt, llm=llm)

# File handlers
def extract_text_from_pdf(file) -> str:
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file) -> str:
    try:
        doc = docx.Document(file)
        return " ".join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

# Updated grammar checker using TextBlob
def highlight_grammar_issues(text: str) -> Tuple[str, List]:
    if not text.strip():
        return "", []
    try:
        blob = TextBlob(text)
        corrected = str(blob.correct())
        corrections = []
        highlighted = text
        if corrected.lower() != text.lower():
            highlighted = (
                f"<span style='background-color: #ffcccc;'>Original: {text}</span><br><br>"
                f"<span style='background-color: #ccffcc;'>Suggested: {corrected}</span>"
            )
            corrections.append({
                'original': text,
                'corrected': corrected
            })
        return highlighted, corrections
    except Exception as e:
        st.error(f"Grammar check failed: {str(e)}")
        return text, []

# Essay input
def get_essay_input() -> str:
    st.subheader("📄 Upload Essay (PDF/DOCX) or Paste Text")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    essay_input = ""
    if uploaded_file:
        with st.spinner("Extracting text from file..."):
            if uploaded_file.name.endswith(".pdf"):
                essay_input = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".docx"):
                essay_input = extract_text_from_docx(uploaded_file)
    essay_input_manual = st.text_area("Or paste your essay here:", height=300)
    if essay_input_manual.strip():
        essay_input = essay_input_manual.strip()
    return essay_input

essay_input = get_essay_input()

# Session state
if "grades" not in st.session_state:
    st.session_state.grades = []

# Score extractor (regex-based)
def extract_score(text: str) -> str:
    match = re.search(r"(?i)(?:Final\s*Score|Score)[^\d]*(\d+(?:\.\d+)?)\s*/\s*(\d+)", text)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return "N/A"

# Evaluation function
def evaluate_essay() -> None:
    if not essay_input.strip():
        st.warning("Please upload or paste an essay first.")
        return

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

    with st.spinner("Checking grammar..."):
        highlighted_text, grammar_matches = highlight_grammar_issues(essay_input)

    st.success("✅ Evaluation complete!")

    st.subheader("📊 Evaluation Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Detected Language**: `{SUPPORTED_LANGUAGES.get(detected_lang, 'English')}`")
    with col2:
        st.markdown(f"**Grammar Suggestions**: `{len(grammar_matches)}`")

    st.subheader("✨ Grammar Suggestions")
    st.markdown(highlighted_text, unsafe_allow_html=True)

    st.subheader("🗣️ Detailed Feedback")
    st.markdown(result)

    score = extract_score(result)

    st.session_state.grades.append({
        "Level": level,
        "Language": SUPPORTED_LANGUAGES.get(detected_lang, "English"),
        "Max Score": max_score,
        "Score": score,
        "Grammar Suggestions": len(grammar_matches),
        "Feedback": result
    })

# Admin Panel functions
def export_grades() -> None:
    if st.session_state.grades:
        df = pd.DataFrame(st.session_state.grades)
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="essay_grades.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.warning("No grades to export yet.")

def clear_grades() -> None:
    st.session_state.grades = []
    st.sidebar.success("Grades cleared!")

# Main logic
if st.button("Evaluate Essay", help="Start the evaluation process"):
    evaluate_essay()

# Admin panel
st.sidebar.header("📁 Admin Panel")
st.sidebar.button("Export Grades", on_click=export_grades)
st.sidebar.button("Clear Grades", on_click=clear_grades)
