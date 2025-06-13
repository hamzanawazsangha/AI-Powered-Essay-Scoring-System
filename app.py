import streamlit as st
from langchain_community.llms import HuggingFaceHub  # updated import
from langchain.chat_models import ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tempfile
import os
import docx
import fitz  # PyMuPDF
import pandas as pd
from langdetect import detect
import language_tool_python
from dotenv import load_dotenv
import re

# Load environment variables (e.g., Hugging Face API token)
load_dotenv()

st.set_page_config(page_title="Essay Grader", layout="wide")
st.title("📝 Automated Essay Grading System (Multilingual + Grammar Highlights)")

# Sidebar configuration
st.sidebar.header("Configuration")
max_score = st.sidebar.selectbox("Maximum Score", [5, 10, 20, 50, 100], index=0)
level = st.sidebar.selectbox("Student Level", ["Beginner", "Intermediate", "Advanced"], index=0)

# Rubric definitions
rubrics = {
    "Beginner": "1. Grammar and Mechanics\n2. Structure\n3. Relevance\n4. Vocabulary\n5. Clarity",
    "Intermediate": "1. Grammar\n2. Structure\n3. Relevance to Topic\n4. Vocabulary\n5. Coherence and Argumentation",
    "Advanced": "1. Grammar and Syntax\n2. Structural Flow\n3. Argument Depth\n4. Lexical Choice\n5. Critical Thinking and Relevance"
}
rubric = rubrics[level]

# Load open-source LLM
llm = ChatHuggingFace(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

# Prompt
prompt = PromptTemplate(
    input_variables=["essay", "criteria", "max_score", "language"],
    template="""
You are an expert essay evaluator fluent in {language}. Evaluate the following essay (written in {language}) using the rubric and give a score out of {max_score}.

Essay:
{essay}

Rubric:
{criteria}

Return:
1. Final Score out of {max_score}
2. Mistakes and why marks were deducted (mention how much per issue)
3. Suggestions to improve the writing
"""
)

chain = LLMChain(prompt=prompt, llm=llm)

# File handlers
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return " ".join(page.get_text() for page in doc)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join(para.text for para in doc.paragraphs)

def highlight_grammar_issues(text):
    try:
        lang = detect(text)
        tool = language_tool_python.LanguageTool(lang)
    except:
        tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(text)
    highlighted = text
    for match in reversed(matches):
        start = match.offset
        end = match.offset + match.errorLength
        suggestion = ", ".join(match.replacements[:2]) if match.replacements else ""
        replacement = f"**[{text[start:end]}](⚠️: {suggestion})**"
        highlighted = highlighted[:start] + replacement + highlighted[end:]
    return highlighted, matches

# Essay input
st.subheader("📄 Upload Essay (PDF/DOCX) or Paste Text")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
essay_input = ""

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        essay_input = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        essay_input = extract_text_from_docx(uploaded_file)

essay_input_manual = st.text_area("Or paste your essay here:", height=300)
if essay_input_manual.strip():
    essay_input = essay_input_manual.strip()

# Grading storage
if "grades" not in st.session_state:
    st.session_state.grades = []

# Evaluation
if st.button("Evaluate Essay"):
    if not essay_input:
        st.warning("Please upload or paste an essay first.")
    else:
        with st.spinner("Detecting language..."):
            try:
                detected_lang = detect(essay_input)
            except:
                detected_lang = "unknown"

        with st.spinner("Evaluating essay..."):
            result = chain.run(
                essay=essay_input,
                criteria=rubric,
                max_score=max_score,
                language=detected_lang
            )

        with st.spinner("Checking grammar..."):
            highlighted_text, grammar_matches = highlight_grammar_issues(essay_input)

        st.success("✅ Evaluation complete!")
        st.subheader("📊 Evaluation Summary")
        st.markdown(f"**Detected Language**: `{detected_lang}`")

        st.markdown("### ✨ Grammar Issues Highlighted")
        st.markdown(highlighted_text, unsafe_allow_html=True)

        st.text_area("🗣️ Feedback", value=result, height=400)

        score_line = [line for line in result.split("\n") if "Score" in line]
        score = score_line[0].split(":")[-1].strip() if score_line else "N/A"

        st.session_state.grades.append({
            "Level": level,
            "Language": detected_lang,
            "Max Score": max_score,
            "Score": score,
            "Feedback": result
        })

# Admin Panel
st.sidebar.header("📁 Admin Panel")
if st.sidebar.button("Export Grades"):
    if st.session_state.grades:
        df = pd.DataFrame(st.session_state.grades)
        csv_path = os.path.join(tempfile.gettempdir(), "essay_grades.csv")
        df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.sidebar.download_button("Download CSV", f, file_name="essay_grades.csv", mime="text/csv")
    else:
        st.sidebar.warning("No grades to export yet.")
