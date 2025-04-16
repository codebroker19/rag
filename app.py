import streamlit as st
import PyPDF2
from ollama import chat
import time
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

def generate_response(user_input):
    # Prepare the message payload for the model
    messages = [{
        "role": 'system', "content": 'You are an AI assistant which writes high industry production level C++ code',
        'role' : 'user',  "content": user_input
    }]
    # Call the Ollama chat API for the Llama 3.2 model
    start_time = time.time()
    
    estimated_time = 10
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(estimated_time, 0, -1):
        status_text.write(f'Generating response in {i}s...')
        #status_text.write(f"⏳ Generating output in: **{remaining} seconds**...")
        progress_bar.progress((estimated_time - i)/estimated_time)
        time.sleep(1)
    
    status_text.write('Generating Response...')
    
    response = chat(model="llama3.2", messages=messages,
                    options={
            "temperature": 0.3,   
            "top_k": 50,         
            "top_p": 0.9,        
            "max_tokens": 300,   
            "repeat_penalty": 1.2  
        })
    # Return the assistant's response content
    progress_bar.progress(1.0)
    end_time = time.time()
    time_taken = round(end_time-start_time, 2)
    return response['message']['content'], time_taken

# Set up the Streamlit app
st.title("Local Chat Application using Llama 3.2 & Ollama")

def chunk_text(text, chunk_size=500, overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# --- Helper: Build vectorstore from text chunks ---
def build_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="llama3.2")
    return FAISS.from_texts(chunks, embeddings)

#####################################################     RAG      #######################################################

def generate_rag_response(vectorstore, user_input):
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""Use the following context to answer the user's question:

{context}

Question: {user_input}"""

    messages = [
        {"role": "system", "content": "You are my assistant which generates industry level C++ code."},
        {"role": "user", "content": prompt}
    ]
    response = chat(model="llama3.2", messages=messages, options = {"temperature": 0.3,
            "top_k": 50,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "num_predict": 400})
    return response['message']['content']



def extract_text_from_pdf(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        import PyPDF2
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        except PyPDF2.errors.PdfReadError:
            st.error("⚠️ Error reading file. It might be corrupted or incomplete.")
            return ""
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        return ""
        
    

# Create a text input box for the user's prompt
user_input = st.text_input("Enter your query:")

upload_file = st.file_uploader('Upload your file', type = ['txt', 'pdf'])

rag_mode = st.checkbox("Use RAG (Retrieval-Augmented Generation)")

# When the user clicks "Send", generate a response

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{
		'role' : 'system', 
        'content': 'You are an AI assistant which writes high insdustry production level C++ code'
	}]

if upload_file:
    if upload_file.type  == 'txt/plain':
        user_input = upload_file.read().decode('utf-8')
    elif upload_file.type == 'application/pdf':
        user_input = extract_text_from_pdf(upload_file)
        

if st.button("Send"):
    if user_input:
        with st.spinner("Generating response..."):
            if rag_mode:
                if upload_file:
                    text = extract_text_from_pdf(upload_file)
                    chunks = chunk_text(text)
                    vectorstore = build_vectorstore(chunks)
                    response_text = generate_rag_response(vectorstore, user_input)
                    response_time = "using RAG"
                else:
                    st.warning("⚠️ You enabled RAG but didn't upload a file. Falling back to normal LLM.")
                    response_text, response_time = generate_response(user_input)
            else:
                response_text, response_time = generate_response(user_input)

        st.text_area("Response", value=response_text, height=200)
        st.write(f"Response generated in **{response_time}**")
