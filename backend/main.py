from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import fitz
import docx
import pytesseract
from PIL import Image
import os

# ---------------- OPENAI CONFIG ---------------- #

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ---------------- FASTAPI APP ---------------- #

app = FastAPI()

# ---------------- CORS ---------------- #

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- TESSERACT PATH ---------------- #

# Uncomment if needed
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

# ---------------- FILE TEXT EXTRACTION ---------------- #

def extract_text(file: UploadFile):

    filename = file.filename.lower()

    # PDF
    if filename.endswith(".pdf"):

        pdf_bytes = file.file.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

        text = ""

        for page in pdf:
            text += page.get_text()

        return text

    # DOCX
    elif filename.endswith(".docx"):

        doc = docx.Document(file.file)

        text = "\n".join(
            [para.text for para in doc.paragraphs]
        )

        return text

    # IMAGE
    elif filename.endswith((".png", ".jpg", ".jpeg")):

        image = Image.open(file.file)

        text = pytesseract.image_to_string(image)

        return text

    # TXT
    elif filename.endswith(".txt"):

        content = file.file.read()

        return content.decode("utf-8")

    return ""

# ---------------- SKILL EXTRACTION ---------------- #

def extract_skills(text):

    words = text.lower().split()

    clean_words = []

    for word in words:

        word = word.strip(",.!?()[]{}<>:;")

        if len(word) > 2:
            clean_words.append(word)

    return list(set(clean_words))

# ---------------- TF-IDF SCORE ---------------- #

def compute_tfidf_score(resume, jd):

    tfidf = TfidfVectorizer(stop_words="english")

    vectors = tfidf.fit_transform([resume, jd])

    score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return score * 100

# ---------------- SEMANTIC SCORE ---------------- #

def compute_semantic_score(resume, jd):

    try:

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[resume, jd]
        )

        emb1 = response.data[0].embedding
        emb2 = response.data[1].embedding

        score = cosine_similarity(
            [emb1],
            [emb2]
        )[0][0]

        return score * 100

    except Exception as e:

        print("Semantic Score Error:", e)

        return 0

# ---------------- ANALYZE API ---------------- #

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):

    try:

        resume_text = extract_text(file)

        if not resume_text.strip():

            return {
                "error": "Could not extract text from resume"
            }

        tfidf_score = compute_tfidf_score(
            resume_text,
            job_description
        )

        semantic_score = compute_semantic_score(
            resume_text,
            job_description
        )

        final_score = (
            (0.6 * tfidf_score) +
            (0.4 * semantic_score)
        )

        resume_skills = extract_skills(resume_text)

        jd_skills = extract_skills(job_description)

        missing_skills = list(
            set(jd_skills) - set(resume_skills)
        )

        return {

            "ATS Score": round(final_score, 2),

            "Keyword Score": round(tfidf_score, 2),

            "Semantic Score": round(semantic_score, 2),

            "Missing Skills": missing_skills[:10]
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# ---------------- HOME ROUTE ---------------- #

@app.get("/")
def home():

    return {
        "message": "ATS Resume Analyzer API is running successfully"
    }
