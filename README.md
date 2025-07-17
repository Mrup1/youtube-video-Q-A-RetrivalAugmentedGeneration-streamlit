# youtube-video-Q-A-RetrivalAugmentedGeneration-streamlit

This Streamlit application provides an interactive way to ask questions about the content of any public YouTube video. It leverages Retrieval-Augmented Generation (RAG) to fetch, process, and query video transcripts, offering contextual answers based solely on the video's spoken content.

Features
YouTube Transcript Extraction: Automatically fetches the full transcript of a provided YouTube video URL.

Contextual Q&A: Ask questions about the video content, and the app will provide answers based only on the information available in the transcript.

Retrieval-Augmented Generation (RAG): Utilizes a RAG pipeline with Hugging Face models (a large language model for generation and embedding models for retrieval) to ensure accurate and context-aware responses.

User-Friendly Interface: Built with Streamlit for an intuitive and easy-to-use web interface.

How it Works
Input URL: The user provides a YouTube video URL.

Transcript Fetching: The app uses the youtube-transcript-api to extract the full transcript of the video.

Text Chunking: The transcript is split into smaller, manageable chunks using RecursiveCharacterTextSplitter.

Vector Store Creation: These chunks are then embedded using a Hugging Face embeddings model (sentence-transformers/all-MiniLM-L6-v2) and stored in a FAISS vector store, creating a searchable knowledge base.

Question Answering: When a user asks a question, the app retrieves the most relevant chunks from the vector store. These chunks, along with the user's question, are then fed to a Hugging Face large language model (mistralai/Mistral-7B-Instruct-v0.2) to generate a concise and contextually accurate answer.

note:you first need to make a hugging face acc to make your huggingfacehub_api_key putt that key in the code i mentioned it where then run the code
and install all the dependencies required for this 

pip install streamlit
pip install youtube-transcript-api
pip install langchain-google-genai
pip install langchain-community
pip install python-dotenv
