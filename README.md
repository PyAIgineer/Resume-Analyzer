# Resume-Analyzer

> **Author:** Sameer Mankar  

Resume Analyzer is a simple resume analyzer that extracts, analyzes, and evaluates resumes using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**. Designed with HR professionals in mind, it provides interactive resume analysis and AI-generated assessments. Simply upload a **PDF resume**, and let the AI do the rest!

## 📌 Features
✅ Extracts text from PDF resumes  
✅ Uses **RAG** for resume analysis  
✅ Categorizes skills into predefined groups (e.g., **Programming, Machine Learning, Data Visualization**)  
✅ Calculates **years of experience** from job history  
✅ Provides an **AI-generated HR assessment**  
✅ Interactive **Q&A** about resumes  
✅ **Streamlit-based web UI** for easy uploads & analysis  

## 🎥 Demo Video
### Watch the video
https://github.com/user-attachments/assets/af2f794c-79a8-44ea-9af2-14e7dcd93ca7

## 🚀 Installation
### Prerequisites
- Python **3.8+**
- **pip**
- Virtual environment (**recommended**)

### Setup
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-repo-name/resume-analyzer.git
   cd resume-analyzer
   ```
2. **Create a virtual environment** (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```
3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## 🎯 Usage
### 🔹 Command Line Mode
Run the Resume Analyzer directly from the terminal:
```sh
python AN.py
```
Follow the prompts to provide the resume file path and interact with the system.

### 🔹 Web App Mode
To launch the **Streamlit-based web interface**:
```sh
streamlit run app.py
```
Upload a **PDF resume** and receive an **AI-generated assessment**.

## 🔑 Environment Variables
This project requires an API key for **Groq’s LLM services**. Set up the environment variable:
```sh
export GROQ_API_KEY="your_api_key_here"
```
On Windows:
```sh
set GROQ_API_KEY="your_api_key_here"
```

## 📂 File Structure
```
.
├── AN.py               # Backend resume analysis logic
├── app.py              # Streamlit-based frontend UI
├── requirements.txt    # Required dependencies
├── README.md           # Project documentation
```

## 📦 Dependencies
- `streamlit` → UI for web-based resume analysis
- `PyMuPDF` → PDF text extraction
- `langchain` & `FAISS` → RAG-based resume processing
- `sentence-transformers` → Embeddings

## 🤝 Contribution
Contributions are always welcome! Feel free to submit **issues** or **pull requests**.

## 📜 License
This project is licensed under the **MIT License**.

---
⭐ **Star this repo** if you found it helpful!
```

