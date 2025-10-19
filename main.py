import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from keybert import KeyBERT
import seaborn

# Parameters

resume_path = "resumes/Bill_Wang_Resume.pdf"
job_desc_path = "descriptions/job_desc.txt"

# DO NOT EDIT

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    return text

def analyze_resume(resume_path, job_description_path):
    

    # Convert job description txt to file
    with open(job_description_path, 'r') as file:
        job_text = file.read().replace('\n', '')

    resume_text = extract_text_from_pdf(resume_path)

    # Extract skills and entities from resume
    doc = nlp(resume_text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "ORG"]]

    # Extract skills and entities from job description
    job_doc = nlp(job_text)
    job_skills = [ent.text for ent in job_doc.ents if ent.label_ in ["SKILL", "ORG"]]


    # Extract keywords
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(resume_text, keyphrase_ngram_range=(1, 2), stop_words='english', use_mmr=True, diversity=0.3)
    job_keywords = kw_model.extract_keywords(job_text, keyphrase_ngram_range=(1, 2), stop_words='english')

    print(keywords)
    print(job_keywords)

    # Calculate similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    # skills_vector = vectorizer.fit_transform([skills, job_skills])
    # skill_match = cosine_similarity(skills_vector[0], skills_vector[1])[0][0]
    
    return [
        skills,
        similarity,
        [(ent.text, ent.label_) for ent in doc.ents]
        # skill_match
    ]

skills, similarity, entities = analyze_resume(resume_path, job_desc_path)
