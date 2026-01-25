from flask import Flask, request, render_template, redirect, session
import pdfplumber
import re
import math
import json
from collections import Counter
from werkzeug.security import check_password_hash

app = Flask(__name__)
app.secret_key = "super-secret-key"

# =========================
# -------- HELPERS --------
# =========================

# ---------- PDF ----------
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.lower() + "\n"
    except Exception as e:
        print("PDF Error:", e)
    return text



# ---------- JSON ----------
def load_admins():
    with open("admins.json") as f:
        return json.load(f)["admins"]

def validate_admin(username, password):
    for admin in load_admins():
        if admin["username"] == username and \
           check_password_hash(admin["password_hash"], password):
            return True
    return False

def load_job():
    with open("job.json") as f:
        return json.load(f)

def save_job(jd, top_n):
    with open("job.json", "w") as f:
        json.dump({"job_description": jd, "top_n": top_n}, f)

def load_submissions():
    with open("submissions.json") as f:
        raw = json.load(f)["submissions"]

    unique = {}
    for sub in raw:
        unique[sub["name"]] = sub["score"]  # keeps latest score

    return [{"name": k, "score": v} for k, v in unique.items()]

def save_submission(name, score):
    # 🔥 SANITIZE name again (last line of defense)
    name = name.replace("Name:", "").replace("NAME:", "").strip()

    data = {"submissions": load_submissions()}

    for sub in data["submissions"]:
        if sub["name"] == name:
            sub["score"] = score
            break
    else:
        data["submissions"].append({
            "name": name,
            "score": score
        })

    with open("submissions.json", "w") as f:
        json.dump(data, f, indent=2)

def get_top_candidates():
    job = load_job()
    subs = load_submissions()
    return sorted(subs, key=lambda x: x["score"], reverse=True)[:job["top_n"]]

def extract_candidate_name(text):
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # 🔥 Clean prefixes
        line = line.replace("Name:", "").replace("NAME:", "").strip()

        # Accept realistic names only
        if 1 < len(line.split()) <= 4:
            return line.title()

    return "Unknown Candidate"

# =========================
# ----- SECTION LOGIC -----
# =========================

def extract_sections(text):
    sections = {"skills": "", "experience": "", "education": ""}

    patterns = {
        "skills": r"skills(.*?)(experience|education|projects|$)",
        "experience": r"experience(.*?)(education|projects|skills|$)",
        "education": r"education(.*?)(experience|projects|skills|$)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.S)
        if match:
            sections[key] = match.group(1).strip()

    return sections



# =========================
# ----- SKILL LOGIC -------
# =========================

SKILL_SET = {
    "python", "java", "c++", "flask", "django", "sql", "mysql",
    "machine learning", "deep learning", "nlp",
    "html", "css", "javascript", "git", "rest api"
}

def extract_skills(text):
    return {skill for skill in SKILL_SET if skill in text}

def skill_match_score(resume_skills, jd_skills):
    if not jd_skills:
        return 0
    return round((len(resume_skills & jd_skills) / len(jd_skills)) * 100, 2)

def evaluate_skills(resume_skills, jd_skills):
    return {
        "matched_skills": list(resume_skills & jd_skills),
        "missing_skills": list(jd_skills - resume_skills)
    }

# =========================
# ---- EXPERIENCE ---------
# =========================

STOPWORDS = {"and", "the", "to", "of", "in", "for", "with"}

def experience_keyword_score(exp_text, jd_text):
    jd_words = {w for w in jd_text.split() if w not in STOPWORDS}
    exp_words = exp_text.split()
    if not jd_words or not exp_words:
        return 0
    matched = [w for w in exp_words if w in jd_words]
    return round((len(matched) / len(jd_words)) * 100, 2)

def evaluate_experience(exp_text, jd_text):
    score = experience_keyword_score(exp_text, jd_text)
    status = "relevant" if score >= 60 else "partially relevant" if score >= 35 else "not relevant"
    return {"experience_score": score, "relevance": status}

# =========================
# ---- EDUCATION ----------
# =========================

def evaluate_education(edu_text):
    keywords = ["b.tech", "m.tech", "bachelor", "master", "engineering", "computer"]
    return {"education_alignment": "aligned" if any(k in edu_text for k in keywords) else "neutral"}

# =========================
# ---- TF-IDF -------------
# =========================

def tfidf_similarity(text1, text2):
    def tf(word, text):
        return text.count(word) / max(len(text), 1)

    def idf(word, docs):
        return math.log(len(docs) / (1 + sum(word in d for d in docs)))

    docs = [text1.split(), text2.split()]
    vocab = set(docs[0]) | set(docs[1])

    v1, v2 = [], []
    for word in vocab:
        v1.append(tf(word, docs[0]) * idf(word, docs))
        v2.append(tf(word, docs[1]) * idf(word, docs))

    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))

    return round((dot / (mag1 * mag2)) * 100, 2) if mag1 and mag2 else 0

def readability_score(text):
    sentences = max(text.count("."), 1)
    words = max(len(text.split()), 1)
    avg = words / sentences

    if avg <= 20:
        return {"readability": "Excellent", "score": 90}
    elif avg <= 30:
        return {"readability": "Good", "score": 70}
    return {"readability": "Poor", "score": 40}

def ats_check(text):
    issues = []

    if "|" in text:
        issues.append("Avoid tables and columns")

    if any(ch in text for ch in ["@", "#", "$"]):
        issues.append("Special characters detected")

    return {
        "ats_friendly": len(issues) == 0,
        "issues": issues
    }

def resume_health(skill_score, readability, ats):
    score = skill_score

    if readability["readability"] == "Excellent":
        score += 10

    if ats["ats_friendly"]:
        score += 10

    return min(score, 100)

# =========================
# ---- FIT MODEL ----------
# =========================

def fit_prediction(skill, exp, tfidf):
    score = round(0.4*skill + 0.35*exp + 0.25*tfidf, 2)
    label = "Good Fit" if score >= 70 else "Moderate Fit" if score >= 45 else "Poor Fit"
    return {"fit_score": score, "fit_label": label}

# =========================
# -------- ROUTES ---------
# =========================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/analyze", methods=["POST"])
def analyze_resume():
    resume_file = request.files.get("resume")
    job = load_job()
    jd_text = job["job_description"].lower()

    resume_text = extract_text_from_pdf(resume_file)
    sections = extract_sections(resume_text)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    skill_score = skill_match_score(resume_skills, jd_skills)
    experience = evaluate_experience(sections["experience"], jd_text)
    tfidf_score = tfidf_similarity(resume_text, jd_text)

    fit = fit_prediction(skill_score, experience["experience_score"], tfidf_score)

    candidate_name = extract_candidate_name(resume_text)
    save_submission(candidate_name, fit["fit_score"])

    readability = readability_score(resume_text)
    ats = ats_check(resume_text)
    health = resume_health(skill_score, readability, ats)

    return render_template(
        "result.html",
        skill_score=skill_score,
        experience_score=experience["experience_score"],
        tfidf_score=tfidf_score,
        fit=fit,
        skills=evaluate_skills(resume_skills, jd_skills),  # ✅ comma added
        readability=readability,
        ats=ats,
        resume_health=health
    )
# ---------- ADMIN ----------

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if validate_admin(request.form["username"], request.form["password"]):
            session["admin"] = True
            return redirect("/admin/dashboard")
        return "Invalid credentials", 401
    return render_template("admin_login.html")

@app.route("/admin/dashboard", methods=["GET", "POST"])
def admin_dashboard():
    if not session.get("admin"):
        return redirect("/admin/login")

    if request.method == "POST":
        save_job(request.form["job_description"], int(request.form["top_n"]))

    return render_template(
        "admin_dashboard.html",
        job=load_job(),
        candidates=get_top_candidates()
    )

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect("/")

if __name__ == "__main__":
    app.run()