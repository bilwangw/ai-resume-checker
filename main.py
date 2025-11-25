import spacy
import numpy as np
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from keybert import KeyBERT
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters

resume_path = "resumes/Bill_Wang_Resume.pdf"
job_desc_path = "descriptions/job_desc.txt"

# \/\/\/\/\/\/ DO NOT EDIT BELOW \/\/\/\/\/\/

# Load NLP model
nlp = spacy.load("en_core_web_lg")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    return text

def analyze_resume(resume_path, job_description_path):
    
    # Convert job description txt to file
    with open(job_description_path, 'r') as file:
        job_text = file.read().replace('\n', '')

    resume_text = extract_text_from_pdf(resume_path)

    # Create nlp object for resume and job description
    doc = nlp(resume_text)
    job_doc = nlp(job_text)

    # Use skillner to extract and compare skills, convert to list of skills
    # remove duplicates and convert to np array by converting -> set -> list -> np.array

    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    resume_annotation = skill_extractor.annotate(resume_text)
    resume_full_matches = resume_annotation["results"]["full_matches"]
    resume_skills = np.array(list(set([match['doc_node_value'] for match in resume_full_matches])))

    job_annotation = skill_extractor.annotate(job_text)
    job_full_matches = job_annotation["results"]["full_matches"]
    job_skills = np.array(list(set([match['doc_node_value'] for match in job_full_matches])))

    job_len = len(job_skills)
    resume_len = len(resume_skills)
    max_len = max(job_len, resume_len)
    min_len = min(job_len, resume_len)

    if resume_len > job_len:
        job_skills = np.pad(job_skills, (0, max_len - min_len), constant_values='-')
    elif resume_len < job_len:
        resume_skills = np.pad(resume_skills, (0, max_len - min_len), constant_values='-')

    skill_df = pd.DataFrame({"Resume" : resume_skills, "Job" : job_skills})
    print(skill_df)
    with open('out.txt', 'w') as output:
        output.write(skill_df.to_string())
    
    # Extract keywords with keyBERT
    kw_model = KeyBERT()
    resume_keywords = kw_model.extract_keywords(resume_text, keyphrase_ngram_range=(1, 2), stop_words='english', use_mmr=True, diversity=0.3, top_n = 10)
    job_keywords = kw_model.extract_keywords(job_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n = 10)

    resume_kw = [x[0] for x in resume_keywords]
    job_kw = [x[0] for x in job_keywords]

    # Calculate similarity
    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([resume_text, job_text])
    text_similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    kw_vectors = vectorizer.fit_transform(["".join(resume_kw), "".join(job_kw)])
    keyword_match = cosine_similarity(kw_vectors[0], kw_vectors[1])[0][0]

    nlp_similarity = doc.similarity(job_doc)

    print(text_similarity, nlp_similarity, keyword_match)

    df = pd.DataFrame({"Similarity Scores" : ["Text", "NLP", "Keywords"], "Values" : [text_similarity, nlp_similarity, keyword_match]})

    sns.barplot(x='Similarity Scores', y='Values', data=df)

    plt.savefig("sns.png")

    return


analyze_resume(resume_path, job_desc_path)
