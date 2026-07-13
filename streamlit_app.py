import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# --- पेज की सेटिंग ---
st.set_page_config(page_title="Syllabus AI - PDF to Q&A", page_icon="📚", layout="wide")
st.title("📚 PDF & Syllabus AI: Question Generator")
st.markdown("अपनी किताब की PDF और सिलेबस डालें, और AI आपके लिए मार्क-वाइज़ सवाल-जवाब तैयार करेगा।")

# --- साइडबार में API की (Key) और इनपुट ---
with st.sidebar:
    st.header("⚙️ सेटिंग्स")
    api_key = st.text_input("अपना Gemini API Key डालें", type="password")
    st.markdown("[यहाँ से फ्री Gemini API Key प्राप्त करें](https://aistudio.google.com/app/apikey)")
    
    st.divider()
    pdf_docs = st.file_uploader("अपनी PDF किताब अपलोड करें", type=["pdf"])
    syllabus_text = st.text_area("अपना सिलेबस यहाँ कॉपी-पेस्ट करें (Topics)")
    process_btn = st.button("प्रोसेस करें और सवाल बनाएँ")

# --- बैकएंड फंक्शन (PDF से टेक्स्ट निकालना) ---
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- टेक्स्ट को छोटे हिस्सों (Chunks) में बांटना ---
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# --- वेक्टर डेटाबेस बनाना ---
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# --- AI से सवाल-जवाब जनरेट करवाना ---
def generate_qa_based_on_syllabus(vector_store, syllabus, api_key):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
    
    # RAG (Retrieval) - सिलेबस से मिलते-जुलते टॉपिक्स खोजना
    docs = vector_store.similarity_search(syllabus, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    तुम एक एक्सपर्ट टीचर हो। नीचे दिए गए 'Syllabus' और 'Book Context' के आधार पर एग्जाम के लिए महत्वपूर्ण सवाल और उनके विस्तृत जवाब तैयार करो।
    ध्यान रहे, कोई भी जानकारी Context से बाहर की नहीं होनी चाहिए।

    Syllabus Topics: {syllabus}
    
    Book Context: {context}

    मुझे आउटपुट इस फॉर्मेट में चाहिए:
    
    ### 1 Mark Questions (Multiple Choice & One Word)
    (कम से कम 3 सवाल और उनके जवाब)

    ### 3 Marks Questions (Short Answer)
    (कम से कम 2 सवाल और उनके सटीक जवाब)

    ### 4 Marks Questions (Concept-based)
    (कम से कम 2 सवाल और उनके जवाब)

    ### 10 Marks Questions (Long & Detailed Answer)
    (कम से कम 1 विस्तृत सवाल और उसका पूरा जवाब)
    """
    
    response = model.invoke(prompt)
    return response.content

# --- मुख्य एग्जीक्यूशन (जब यूज़र बटन दबाए) ---
if process_btn:
    if not api_key:
        st.error("कृपया अपना API Key दर्ज करें!")
    elif not pdf_docs:
        st.error("कृपया एक PDF फाइल अपलोड करें!")
    elif not syllabus_text:
        st.error("कृपया अपना सिलेबस दर्ज करें!")
    else:
        with st.spinner("AI आपकी किताब पढ़ रहा है और सवाल तैयार कर रहा है... इसमें कुछ समय लग सकता है⏳"):
            try:
                # 1. PDF से टेक्स्ट निकालें
                raw_text = get_pdf_text(pdf_docs)
                
                # 2. टेक्स्ट के टुकड़े करें
                text_chunks = get_text_chunks(raw_text)
                
                # 3. डेटाबेस (FAISS) बनाएँ
                vector_store = get_vector_store(text_chunks, api_key)
                
                # 4. AI से आउटपुट लें
                final_result = generate_qa_based_on_syllabus(vector_store, syllabus_text, api_key)
                
                # स्क्रीन पर दिखाएं
                st.success("सफलतापूर्वक तैयार हो गया!")
                st.markdown(final_result)
                
                # डाउनलोड बटन
                st.download_button(
                    label="📥 इसे Text File के रूप में डाउनलोड करें",
                    data=final_result,
                    file_name="generated_qa.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"कुछ खराबी आ गई: {e}")
