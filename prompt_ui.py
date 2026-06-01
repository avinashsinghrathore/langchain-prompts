from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro", task="text-generation")

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

# Static prompts example
# user_input = st.text_input("Enter your prompt")

# Dynamic prompts example
paper_input = st.selectbox(
    "Select Research paper Name",
    [
        "Attention is all you need",
        "BERT:Pre-training of deep bidirectional transformers",
        "GPT-3: Language Models are few-shot learners",
        "Diffusion models beat GANs on image synthesis",
    ],
)

style_input = st.selectbox(
    "Select explanation style",
    ["Beginner-friendly", "Technical", "Code-oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation length",
    [
        "short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)

# Template
template = PromptTemplate(
    template="""Please summarise the research paper titled “{paper_input} with the following specifications:
Explanation style: {style_input}
Explanation length: {length_input}
1.  Mathematical details:
	- Include relevant mathematical equations if present in the paper.
	- Explain the mathematical concepts using simple intuitive code snippets where applicable.
2.  Analogies:
	- use relatable analogies to simplify complex ideas
	if certain information is not available in the paper respond with insufficient information available instead of guessing 
Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
    input_variables=["paper_input", "style_input", "length_input"],
)

# Fill the placeholders
prompt = template.invoke(
    {
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
    }
)

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)
