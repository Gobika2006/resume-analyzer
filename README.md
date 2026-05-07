# 🚀 AI Resume Analyzer

AI-powered ATS Resume Analyzer using:

- FastAPI
- OpenAI Embeddings
- TF-IDF
- OCR
- HTML/CSS Frontend

---

# Features

✅ ATS Resume Score  
✅ Semantic Matching  
✅ Keyword Matching  
✅ Missing Skills Detection  
✅ PDF/DOCX/Image Resume Support

---

# Run Backend

```bash
cd backend

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

$env:OPENAI_API_KEY="YOUR_API_KEY"

uvicorn main:app --reload
