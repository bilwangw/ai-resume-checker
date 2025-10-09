import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

# Parameters

resume_path = "resumes/Bill_Wang_Resume.pdf"
job_desc = "descriptions/job_desc.txt"

# DO NOT EDIT

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    return text

def analyze_resume(resume_text, job_description):
    # Extract skills, entities
    doc = nlp(resume_text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "ORG"]]
    
    # Calculate similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    
    return {
        "skills": skills,
        "similarity_score": similarity,
        "entities": [(ent.text, ent.label_) for ent in doc.ents]
    }

pdf_text = extract_text_from_pdf(resume_path)

print(pdf_text)

analysis = analyze_resume(pdf_text, "job software")
print(analysis)