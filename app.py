import streamlit as st
import pdfplumber
import re
import pandas as pd
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
import torch
import gc

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = ""

def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def setup_cuda():
    """Setup and verify CUDA configuration"""
    if torch.cuda.is_available():
        # Set CUDA device
        torch.cuda.set_device(0)  # Use first GPU
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get device info
        device_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2    # Convert to MB
        
        return {
            "status": True,
            "device": "cuda",
            "name": device_name,
            "memory_allocated": f"{memory_allocated:.2f} MB",
            "memory_reserved": f"{memory_reserved:.2f} MB"
        }
    return {"status": False, "device": "cpu"}

@st.cache_resource
def init_model():
    """Initialize model with CUDA optimization if available"""
    try:
        # Setup CUDA configuration
        cuda_info = setup_cuda()
        device = cuda_info["device"]
        
        # Initialize model with CUDA configuration
        model = Ollama(
            model="llama3.2",
            temperature=0.7,
            device=device,
            gpu_layers=43 if device == "cuda" else 0  # Use GPU layers when CUDA is available
        )
        
        # Display CUDA information
        # if cuda_info["status"]:
        #     st.sidebar.write("### CUDA Configuration")
        #     st.sidebar.write(f"🖥️ GPU: {cuda_info['name']}")
        #     st.sidebar.write(f"💾 Memory Allocated: {cuda_info['memory_allocated']}")
        #     st.sidebar.write(f"💾 Memory Reserved: {cuda_info['memory_reserved']}")
        # else:
        #     st.sidebar.warning("⚠️ Running on CPU - CUDA not available")
            
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def process_chat(question, context):
    """Process chat with CUDA-optimized model"""
    model = init_model()
    if not model:
        return "Model initialization failed. Please check CUDA configuration."
        
    try:
        # Set memory efficient attention when using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache before processing
            
        prompt = f"""As a financial analysis assistant, please help answer this question 
        based on the following context. If the information isn't in the context, say so clearly. 
        And you only need to answer the question briefly with only one language (indonesian or english).
        
        Context: {context}
        
        Question: {question}
        """

        response = model.invoke(prompt)
        
        # Clear CUDA cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return response
    except Exception as e:
        return f"Error processing chat: {str(e)}"

def chat_interface():
    st.sidebar.write("## Abinala Chatbot")
    
    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

    # Chat input
    if question := st.sidebar.text_area("Masukkan pertanyaan Anda di sini", key="chat_input"):
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=question))
        
        # Get response from local model
        with st.spinner('Processing your question...'):
            response = process_chat(question, st.session_state.pdf_content)
        
        # Add assistant message to chat history
        st.session_state.messages.append(AIMessage(content=response))
        
        # Rerun to update chat display
        st.rerun()

st.set_page_config(page_title="Abhinala", page_icon="📈", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def main_page():
    chat_interface()
    # st.sidebar.write("## Abinala Chatbot")
    # st.sidebar.text_area("Masukkan pertanyaan Anda di sini", height=700, key="chat_input", value="")
    st.sidebar.button("Kirim")
    st.sidebar.write("© Licensed by Otoritas Jasa Keuangan 2024")

    # Add CUDA information section in sidebar
    st.sidebar.write("### System Information")
    
    try:
        cuda_info = setup_cuda()
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # Convert to MB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2    # Convert to MB
            
            # Display CUDA information
            st.sidebar.write("🖥️ **GPU Status:** Active")
            st.sidebar.write(f"🎯 **Device:** {device_name}")
            st.sidebar.write(f"💾 **Memory Allocated:** {memory_allocated:.2f} MB")
            st.sidebar.write(f"💾 **Memory Reserved:** {memory_reserved:.2f} MB")
        else:
            st.sidebar.warning("⚠️ Running on CPU - CUDA not available")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.sidebar.write(f"Device: {device.upper()}")
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
    return None
    
    

def extract_data(pdf_file, patterns):
    data = {key: {"Nilai": "Tidak ditemukan", "Status": False, "Halaman": None} for key in patterns}
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            for key, regex in patterns.items():
                if not data[key]["Status"]:
                    match = re.search(regex, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        data[key] = {"Nilai": match.group(1), "Status": True, "Halaman": str(page_num)}
    return data

def extract_financial_ratios(pdf_file):
    ratios = {
        "ROA": {"Nilai": "Tidak ditemukan", "Status": False, "Halaman": None},
        "ROI": {"Nilai": "Tidak ditemukan", "Status": False, "Halaman": None},
        "Debt to Asset Ratio": {"Nilai": "Tidak ditemukan", "Status": False, "Halaman": None}
    }
    
    def clean_number(text):
        """Convert string number to float, handling different number formats."""
        return float(text.replace(".", "").replace(",", ""))

    def calculate_ratio(numerator, denominator):
        """Calculate percentage ratio with rounding."""
        return round((numerator / denominator) * 100, 2)

    def update_ratio(ratio_name, value, page_num):
        """Update ratio dictionary with new values."""
        ratios[ratio_name] = {
            "Nilai": f"{value}%",
            "Status": True,
            "Halaman": page_num
        }

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            # Extract ROA
            if not ratios["ROA"]["Status"]:
                roa_patterns = [
                    r'laba bersih\s*(?:\([^)]+\))?.*?([\d.,]+).*?total aset\s*(?:\([^)]+\))?.*?([\d.,]+)'               
                ]
                for pattern in roa_patterns:
                    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        try:
                            laba_rugi_bersih = clean_number(match.group(1))
                            total_asset = clean_number(match.group(2))
                            roa = calculate_ratio(laba_rugi_bersih, total_asset)
                            update_ratio("ROA", roa, page_num)
                            break
                        except ValueError:
                            continue

            # Extract ROI
            if not ratios["ROI"]["Status"]:
                match = re.search(r'LABA BERSIH.*?([\d.,]+).*?TOTAL EKUITAS.*?([\d.,]+)', 
                                text, re.DOTALL)
                if match:
                    try:
                        laba_bersih = clean_number(match.group(1))
                        total_equity = clean_number(match.group(2))
                        roi = calculate_ratio(laba_bersih, total_equity)
                        update_ratio("ROI", roi, page_num)
                    except ValueError:
                        continue

            # Extract Debt to Asset Ratio
            if not ratios["Debt to Asset Ratio"]["Status"]:
                match = re.search(r'\bTOTAL LIABILITAS\b\s*([\d.,]+).*?\bTOTAL ASET\b\s*([\d.,]+)', 
                                text, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        total_debt = clean_number(match.group(1))
                        total_asset = clean_number(match.group(2))
                        dar = calculate_ratio(total_debt, total_asset)
                        update_ratio("Debt to Asset Ratio", dar, page_num)
                    except ValueError:
                        continue

    return ratios

def extract_directors_data(pdf_file):
    directors = []
    pattern = r"(?<=Name\s)(.*?)(?=\s*Position\s+)(.*?)\s*(?=Tenure Start Date)"

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            matches = re.findall(pattern, text, re.DOTALL)

            for names, positions in matches:
                # Split names into two-word groups
                name_list = [name.strip() for name in re.split(r'\s{2,}', names) if name]
                split_names = []
                for name in name_list:
                    words = name.split()
                    for i in range(0, len(words), 2):
                        split_names.append(' '.join(words[i:i + 2]))

                # Process positions into two-word groups
                position_list = [pos.strip() for pos in re.split(r'\s{2,}', positions) if pos]
                split_positions = []
                for pos in position_list:
                    words = pos.split()
                    for i in range(0, len(words), 2):
                        split_positions.append(' '.join(words[i:i + 2]))

                # Align names with positions
                for i in range(min(len(split_names), len(split_positions))):
                    directors.append({
                        "Name": split_names[i],
                        "Position": split_positions[i],
                        "Halaman": page_num,
                        "Status": "Ditemukan"
                    })

    if not directors:
        directors.append({
            "Name": "Tidak ditemukan",
            "Position": "Tidak ditemukan",
            "Halaman": None,
            "Status": "Tidak Ditemukan"
        })

    return directors


def display_extracted_data(data, title, unit=""):
    st.header(f"Hasil Ekstraksi Data {title}")
    summary = [
        {
            "Keterangan": k,
            "Nilai": f"{v['Nilai']} {unit}" if unit and v["Status"] else v["Nilai"],
            "Status": "Ditemukan" if v["Status"] else "Tidak Ditemukan",
            "Halaman": v["Halaman"]
        }
        for k, v in data.items()
    ]
    st.dataframe(pd.DataFrame(summary).style.set_properties(**{'max-height': '400px', 'overflow': 'auto'}), use_container_width=True)

def display_directors_data(directors):
    st.header("Hasil Ekstraksi Data Direktur dan Komisaris")
    st.dataframe(pd.DataFrame(directors).style.set_properties(**{'max-height': '400px', 'overflow': 'auto'}), use_container_width=True)

def display_missing_data(data):
    missing_data = []  
    if isinstance(data, list):
        # For directors, check each entry's status
        missing_data = [director for director in data if director["Status"] == "Tidak Ditemukan"]
    else:
        # For other data types (dictionary)
        missing_data = [
            {"Keterangan": k, "Nilai": v["Nilai"], "Status": v["Status"], "Halaman": v["Halaman"]}
            for k, v in data.items() if not v["Status"]
        ]

    # Check if any missing data was found
    if missing_data:
        st.header("Data yang Tidak Ditemukan")
        st.dataframe(pd.DataFrame(missing_data), use_container_width=True)
    else:
        st.success("Semua data berhasil ditemukan dan tidak ada masalah dalam laporan.")

financial_patterns = {
    "Total Ekuitas": r"(?i)\b(?:TOTAL EKUITAS|JUMLAH EKUITAS)\b\s*([\d.,]+)",
    "Total Liabilitas": r"(?i)\b(?:TOTAL LIABILITAS|JUMLAH LIABILITAS)\b\s*([\d.,]+)",
    "Total Pendapatan": r"(?i)(?:Total Pendapatan|Jumlah Pendapatan | PENDAPATAN USAHA | Pendapatan Neto).*?([\d.,]+)",
    "Laba Bruto": r"(?i)LABA BRUTO.*?([\d.,]+)",
    "Laba Operasional": r"(?i)LABA OPERASIONAL.*?([\d.,]+)"
}

sustainable_patterns = {
    "Scope 1": r"SCOPE 1.*?EMISSION\s*(\d[\d\s,]*\.\d+|\d[\d\s,]*)\s*(?!%)",
    "Scope 2": r"SCOPE 2.*?EMISSION\s*(\d[\d\s,]*\.\d+|\d[\d\s,]*)\s*(?!%)",
    "Scope 3": r"SCOPE 3.*?(?:FINANCED EMISSIONS).*?([\d,]+(?:\.\d+)?)"

}

st.title("Ekstraksi Data dari Laporan Tahunan dan Laporan Keberlanjutan")
st.write("Unggah file PDF laporan untuk ekstraksi data.")

uploaded_files = {
    "Keuangan": st.file_uploader("Pilih file PDF untuk data keuangan", type="pdf", key="financial"),
    "Keberlanjutan": st.file_uploader("Pilih file PDF untuk data keberlanjutan", type="pdf", key="sustainable"),
    "Direktur": st.file_uploader("Pilih file PDF untuk data direktur dan komisaris", type="pdf", key="directors")
}

extracted_data = {}

# Extract and display financial data
if uploaded_files["Keuangan"]:
    financial_data = extract_data(uploaded_files["Keuangan"], financial_patterns)
    financial_ratios = extract_financial_ratios(uploaded_files["Keuangan"])
    combined_financial_data = {**financial_data, **financial_ratios}
    extracted_data["Keuangan"] = combined_financial_data
    display_extracted_data(combined_financial_data, "Keuangan")

    st.session_state.pdf_content += extract_pdf_text(uploaded_files["Keuangan"])


# Extract and display sustainable data
if uploaded_files["Keberlanjutan"]:
    sustainable_data = extract_data(uploaded_files["Keberlanjutan"], sustainable_patterns)
    extracted_data["Keberlanjutan"] = sustainable_data
    display_extracted_data(sustainable_data, "Keberlanjutan", unit="ton CO₂e")

    st.session_state.pdf_content += extract_pdf_text(uploaded_files["Keberlanjutan"])


# Extract and display directors data
if uploaded_files["Direktur"]:
    directors_data = extract_directors_data(uploaded_files["Direktur"])
    extracted_data["Direktur"] = directors_data
    display_directors_data(directors_data)

    st.session_state.pdf_content += extract_pdf_text(uploaded_files["Direktur"])


# Show Generate Report button if any file is uploaded
if uploaded_files["Keuangan"] or uploaded_files["Keberlanjutan"] or uploaded_files["Direktur"]:
    if st.button("Generate Report"):
        # Only display missing data for successfully extracted documents
        for section, data in extracted_data.items():
            display_missing_data(data)

if st.session_state.logged_in:
    main_page()
else:
    main_page()