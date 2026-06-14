import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# -------------------------------------------------
# Environment Variables
# -------------------------------------------------

os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Replace with your Groq API key
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------

st.set_page_config(
    page_title="Zyro HR Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Zyro Dynamics HR Assistant")

st.markdown(
    "Ask questions related to company HR policies, benefits, leaves, payroll, travel, and employee guidelines."
)

# -------------------------------------------------
# Load & Cache Documents
# -------------------------------------------------

@st.cache_resource
def load_rag_pipeline():

    CORPUS_PATH = "hr_pdfs"

    loader = PyPDFDirectoryLoader(CORPUS_PATH)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever


retriever = load_rag_pipeline()

# -------------------------------------------------
# LLM
# -------------------------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -------------------------------------------------
# Prompt
# -------------------------------------------------

prompt_template = """
You are an HR assistant for Zyro Dynamics.

Answer ONLY from the provided HR policy context.

If the answer is not found in the context, say:
"I could not find this information in the HR policy documents."

Keep answers professional and concise.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# -------------------------------------------------
# HR Guardrails
# -------------------------------------------------

HR_KEYWORDS = [
    "leave",
    "salary",
    "payroll",
    "benefits",
    "attendance",
    "holiday",
    "remote",
    "policy",
    "employee",
    "insurance",
    "travel",
    "reimbursement",
    "promotion",
    "resignation",
    "termination",
    "maternity",
    "paternity",
    "hr",
    "vacation",
    "sick leave",
    "work from home"
]

REFUSAL_MESSAGE = (
    "I'm only able to answer HR-related questions "
    "based on the company policy documents."
)

def is_hr_question(question):

    question = question.lower()

    return any(keyword in question for keyword in HR_KEYWORDS)

# -------------------------------------------------
# Ask Function
# -------------------------------------------------

def ask_hr_bot(question):

    if not is_hr_question(question):
        return REFUSAL_MESSAGE, []

    docs = retriever.invoke(question)

    if len(docs) == 0:
        return (
            "I could not find this information in the HR policy documents.",
            []
        )

    context = "\n\n".join([doc.page_content for doc in docs])

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    sources = []

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")

        if source not in sources:
            sources.append(source)

    return response.content, sources

# -------------------------------------------------
# Chat History
# -------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------
# Chat Input
# -------------------------------------------------

user_question = st.chat_input("Ask an HR question...")

if user_question:

    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        with st.spinner("Searching HR policies..."):

            answer, sources = ask_hr_bot(user_question)

            st.markdown(answer)

            if sources:

                st.markdown("### 📚 Sources")

                for src in sources:
                    st.markdown(f"- `{src}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
