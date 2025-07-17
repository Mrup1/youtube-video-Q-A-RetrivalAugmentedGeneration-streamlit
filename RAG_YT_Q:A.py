
import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Set your Hugging Face API token
# It's recommended to set this as an environment variable (e.g., in .env file or system-wide)
# For example: export HUGGINGFACEHUB_API_TOKEN="hf_YOUR_TOKEN_HERE"
# If not set as an environment variable, uncomment the line below and replace with your token:
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_YOUR_TOKEN_HERE"

# --- RAG Pipeline Components ---

@st.cache_resource
def load_llm():
    """Loads and caches the HuggingFace LLM."""
    try:
        llm_huggingface = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            temperature=0.1,
            max_new_tokens=200, # Increased max_new_tokens for potentially longer answers
            timeout=120
        )
        return ChatHuggingFace(llm=llm_huggingface)
    except Exception as e:
        st.error(f"Failed to load LLM. Please ensure your HUGGINGFACEHUB_API_TOKEN is correctly set and the model is accessible. Error: {e}")
        return None

@st.cache_resource
def load_embeddings():
    """Loads and caches the HuggingFace embeddings model."""
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embeddings model. Error: {e}")
        return None

def get_youtube_transcript(video_id):
    """Fetches the transcript for a given YouTube video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(item["text"] for item in transcript_list)
        return transcript
    except TranscriptsDisabled:
        st.warning("Captions are disabled or not available for this video.")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching the transcript: {e}")
        return None

def process_transcript_and_create_vector_store(transcript_text, embeddings_model):
    """Splits transcript into chunks and creates a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    return vector_store

def format_doc(retrieved_docs):
    """Formats retrieved documents into a single string for the prompt context."""
    return " ".join(doc.page_content for doc in retrieved_docs)

# --- Streamlit App Layout ---

st.set_page_config(page_title="YouTube Video Q&A", layout="centered")

st.title("🎬 YouTube Video Q&A Assistant")
st.markdown("""
    Enter a YouTube video URL, and I'll fetch its transcript.
    Then, you can ask questions about the video content!
""")

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'transcript_fetched' not in st.session_state:
    st.session_state.transcript_fetched = False
if 'llm' not in st.session_state:
    st.session_state.llm = load_llm()
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = load_embeddings()

youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=ShYKkPPhOoc")

# Process video button
if st.button("Process Video"):
    if not youtube_url:
        st.warning("Please enter a YouTube video URL.")
    elif st.session_state.llm is None or st.session_state.embeddings is None:
        st.error("LLM or Embeddings model failed to load. Please check the setup.")
    else:
        with st.spinner("Fetching and processing transcript..."):
            try:
                # Extract video ID from URL
                video_id = youtube_url.split("v=")[-1].split("&")[0]
                transcript_text = get_youtube_transcript(video_id)

                if transcript_text:
                    st.session_state.vector_store = process_transcript_and_create_vector_store(
                        transcript_text, st.session_state.embeddings
                    )
                    st.session_state.transcript_fetched = True
                    st.success("Transcript processed successfully! You can now ask questions.")
                    # Optionally display a snippet of the transcript
                    # st.expander("View Transcript Snippet").write(transcript_text[:500] + "...")
                else:
                    st.session_state.transcript_fetched = False
                    st.session_state.vector_store = None

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.transcript_fetched = False
                st.session_state.vector_store = None

# Q&A Section
if st.session_state.transcript_fetched and st.session_state.vector_store:
    st.markdown("---")
    st.subheader("Ask a Question about the Video")

    user_question = st.text_input("Your Question:", placeholder="e.g., What is discussed about AI jobs?")

    if st.button("Get Answer"):
        if not user_question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Finding answer..."):
                try:
                    # Create retriever
                    retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

                    # Define the prompt template
                    qa_prompt = PromptTemplate(
                        template="""
                            You are a helpful AI assistant.
                            Answer only from the provided transcript context.
                            If the context is insufficient, just say you don't know.

                            Context: {context}
                            Question: {question}
                        """,
                        input_variables=['context', 'question']
                    )

                    # Build the RAG chain
                    rag_chain = (
                        RunnableParallel({
                            "context": retriever | RunnableLambda(format_doc),
                            "question": RunnablePassthrough()
                        })
                        | qa_prompt
                        | st.session_state.llm
                        | StrOutputParser()
                    )

                    # Invoke the chain with the user's question
                    answer = rag_chain.invoke(user_question)
                    st.write("---")
                    st.info(f"**Answer:** {answer}")

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
else:
    if youtube_url and not st.session_state.transcript_fetched:
        st.info("Enter a YouTube URL and click 'Process Video' to start.")
    elif not youtube_url:
        st.info("Enter a YouTube URL to begin.")