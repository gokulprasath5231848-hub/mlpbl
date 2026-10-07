"""
AI Powered Resume Screening and Ranking System — Core Engine

Scoring (unchanged from the Review-II prototype):
  * a plain TF-IDF baseline score, and
  * a synonym-normalised "hybrid" TF-IDF score,
  both computed for every resume against the same job description, plus an
  exact per-term explainability decomposition of the hybrid score.

Added in Iteration 4:
  * extract_skills_from_jd()  — derive the required-skill checklist from the
    job description itself instead of using one hard-coded list.
  * pool_insights()           — skill-gap rollup across the whole candidate pool.
  * justification()           — a one-line, human-readable reason per candidate.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# Skill lexicon: canonical skill -> the surface forms that mean the same
# thing. Used for three separate jobs: synonym normalisation before
# vectorisation (the hybrid step), checklist matching, and extracting the
# required-skill list from a job description.
# ---------------------------------------------------------------------
SKILL_SYNONYMS = {
    # --- data / ML ---
    "python": ["python"],
    "machine learning": ["machine learning", "ml"],
    "natural language processing": ["natural language processing", "nlp"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "deep learning": ["deep learning", "dl"],
    "computer vision": ["computer vision", "opencv"],
    "text classification": ["text classification"],
    "data analysis": ["data analysis", "data analytics"],
    "statistics": ["statistics", "statistical modelling", "statistical modeling"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    # --- data platform ---
    "sql": ["sql", "structured query language"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel", "spreadsheets"],
    # --- engineering ---
    "java": ["java"],
    "spring boot": ["spring boot", "springboot"],
    "javascript": ["javascript", "js"],
    "react": ["react", "react.js", "reactjs"],
    "node.js": ["node.js", "nodejs", "node js"],
    "rest api": ["rest api", "rest apis", "restful api", "restful apis"],
    "html/css": ["html/css", "html and css", "html", "css"],
    # --- platform / tooling ---
    # The spelled-out provider names were missing until the 10,000-resume
    # evaluation measured cloud recall at 0.505: every resume writing
    # "Amazon Web Services" in full was being missed (Section 6.2).
    "cloud": ["cloud", "aws", "amazon web services", "azure", "microsoft azure",
              "gcp", "google cloud", "google cloud platform"],
    "docker": ["docker", "containerisation", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "version control"],
    "linux": ["linux", "unix"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration"],
}

# Skill domains, used only for the radar chart. Every canonical skill in
# SKILL_SYNONYMS belongs to exactly one domain, so a candidate's coverage can
# be read per area of competence instead of as one flat fraction. The grouping
# follows the section comments in SKILL_SYNONYMS above; it changes nothing in
# the scoring, which still works on individual skills.
SKILL_DOMAINS = {
    "ML & AI": ["machine learning", "artificial intelligence", "deep learning",
                "computer vision", "natural language processing",
                "text classification"],
    "Data & Stats": ["data analysis", "statistics", "pandas", "numpy",
                     "excel", "power bi", "tableau"],
    "ML Tooling": ["scikit-learn", "tensorflow", "pytorch"],
    "Databases": ["sql", "postgresql", "mongodb"],
    "Programming": ["python", "java", "javascript", "spring boot", "react",
                    "node.js", "rest api", "html/css"],
    "Cloud & DevOps": ["cloud", "docker", "kubernetes", "git", "linux", "ci/cd"],
}


def domain_of(skill: str) -> str:
    for domain, skills in SKILL_DOMAINS.items():
        if skill in skills:
            return domain
    return "Other"


def domain_coverage(matched, required):
    """Per-domain coverage for the radar chart.

    Only domains the job description actually asks for are returned, so the
    chart never shows an axis the candidate was never required to satisfy.
    Each entry carries the raw counts as well as the percentage, so the axis
    label can state 2/3 rather than only 67%.
    """
    buckets = {}
    for skill in required:
        buckets.setdefault(domain_of(skill), {"required": 0, "matched": 0})
        buckets[domain_of(skill)]["required"] += 1
    matched_set = set(matched)
    for skill in required:
        if skill in matched_set:
            buckets[domain_of(skill)]["matched"] += 1
    out = []
    for domain in SKILL_DOMAINS:          # stable, declared order
        if domain in buckets:
            b = buckets[domain]
            out.append({
                "domain": domain,
                "required": b["required"],
                "matched": b["matched"],
                "pct": round(b["matched"] / b["required"] * 100, 1),
            })
    if "Other" in buckets:
        b = buckets["Other"]
        out.append({"domain": "Other", "required": b["required"],
                    "matched": b["matched"],
                    "pct": round(b["matched"] / b["required"] * 100, 1)})
    return out


# Cap on the resume text carried into the page for the evidence highlighter.
# A review session holds every candidate's text in the browser, so this keeps
# the page from growing without bound on a large batch.
EVIDENCE_CHAR_LIMIT = 8000


def skill_evidence(resume_text: str, required_skills):
    """Locate every surface form of every required skill inside the resume.

    Returns the (possibly truncated) resume text plus a list of non-overlapping
    spans: {skill, surface, start, end, via_synonym}. `via_synonym` is True when
    the matched wording differs from the canonical skill name — that is exactly
    the case plain keyword matching would have missed, so the interface can say
    which synonym earned the credit.
    """
    text = resume_text or ""
    truncated = len(text) > EVIDENCE_CHAR_LIMIT
    if truncated:
        text = text[:EVIDENCE_CHAR_LIMIT]
    lower = text.lower()

    spans = []
    for skill in required_skills:
        for syn in SKILL_SYNONYMS.get(skill, [skill]):
            for m in re.finditer(r"\b" + re.escape(syn) + r"\b", lower):
                spans.append({
                    "skill": skill,
                    "surface": text[m.start():m.end()],
                    "start": m.start(),
                    "end": m.end(),
                    "via_synonym": syn != skill,
                })

    # Longer matches win, so "amazon web services" is marked once rather than
    # leaving a stray fragment behind; ties resolve by position.
    spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    kept, last_end = [], -1
    for sp in spans:
        if sp["start"] >= last_end:
            kept.append(sp)
            last_end = sp["end"]
    return {"text": text, "spans": kept, "truncated": truncated}


# Fallback checklist, used only when a job description yields no recognised
# skills (e.g. a very short or unusual JD).
DEFAULT_REQUIRED_SKILLS = [
    "python", "machine learning", "natural language processing",
    "artificial intelligence", "data analysis", "sql", "cloud",
    "scikit-learn", "tensorflow", "pytorch", "text classification",
]

# Coverage thresholds for the verdict labels. Exposed so the UI can show
# the same numbers it is filtering on.
STRONG_THRESHOLD = 0.6
MODERATE_THRESHOLD = 0.3


def normalize_text(text: str) -> str:
    """Replace synonyms with a canonical skill token so TF-IDF treats
    'ML' and 'Machine Learning' as the same term (the hybrid step)."""
    text_lower = (text or "").lower()
    for canonical, synonyms in SKILL_SYNONYMS.items():
        for syn in synonyms:
            text_lower = re.sub(
                r"\b" + re.escape(syn) + r"\b",
                canonical.replace(" ", "_"),
                text_lower,
            )
    return text_lower


def skill_present(text_lower: str, skill: str) -> bool:
    """True if any surface form of `skill` occurs in already-lowercased text."""
    for syn in SKILL_SYNONYMS.get(skill, [skill]):
        if re.search(r"\b" + re.escape(syn) + r"\b", text_lower):
            return True
    return False


def extract_skills_from_jd(job_desc_text: str):
    """Derive the required-skill checklist from the job description.

    Scans the JD for every canonical skill in the lexicon and returns the
    ones it finds, ordered by where they first appear in the text, so the
    checklist reads in the same order a recruiter wrote it. Falls back to
    DEFAULT_REQUIRED_SKILLS when nothing is recognised.
    """
    text_lower = (job_desc_text or "").lower()
    found = []
    for skill in SKILL_SYNONYMS:
        positions = [
            m.start()
            for syn in SKILL_SYNONYMS[skill]
            for m in re.finditer(r"\b" + re.escape(syn) + r"\b", text_lower)
        ]
        if positions:
            found.append((min(positions), skill))
    if not found:
        return list(DEFAULT_REQUIRED_SKILLS)
    found.sort()
    return [skill for _, skill in found]


def extract_matched_skills(resume_text: str, required_skills=None):
    required_skills = required_skills or DEFAULT_REQUIRED_SKILLS
    text_lower = (resume_text or "").lower()
    matched, missing = [], []
    for skill in required_skills:
        (matched if skill_present(text_lower, skill) else missing).append(skill)
    return matched, missing


def verdict_for_coverage(coverage: float) -> str:
    if coverage >= STRONG_THRESHOLD:
        return "STRONG MATCH"
    elif coverage >= MODERATE_THRESHOLD:
        return "MODERATE MATCH"
    return "WEAK MATCH"


def justification(record: dict) -> str:
    """One-line, human-readable reason for a candidate's placement.

    Built only from values the engine already computed, so the sentence
    can never disagree with the score or the checklist beside it.
    """
    matched = record["Matched_Count"]
    total = record["Total_Skills"]
    missing = record["Missing_Skills"]
    terms = [t["term"] for t in record["Top_Contributing_Terms"][:2]]

    lead = {
        "STRONG MATCH": f"Covers {matched} of {total} required skills",
        "MODERATE MATCH": f"Covers {matched} of {total} required skills",
        "WEAK MATCH": f"Covers only {matched} of {total} required skills",
    }[record["Verdict"]]

    parts = [lead]
    if terms:
        parts.append("strongest overlap on " + " and ".join(terms))
    if missing:
        shown = ", ".join(missing[:3])
        more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        parts.append(f"no evidence of {shown}{more}")
    return "; ".join(parts) + "."


def pool_insights(results, required_skills):
    """Skill-gap rollup across the whole pool.

    Transposes the per-candidate checklist into a per-skill view: for each
    required skill, how many candidates in this pool actually have it.
    Sorted by the largest gap first, because that is the number a recruiter
    acts on (either relax the requirement or widen the search).
    """
    total = len(results)
    if not total:
        return []
    rows = []
    for skill in required_skills:
        have = sum(1 for r in results if skill in r["Matched_Skills"])
        rows.append({
            "skill": skill,
            "have": have,
            "missing": total - have,
            "pct": round(have / total * 100),
        })
    rows.sort(key=lambda r: (r["have"], r["skill"]))
    return rows


def screen_resumes(job_desc_text: str, resumes_df: pd.DataFrame,
                   required_skills=None, with_evidence: bool = False) -> pd.DataFrame:
    """
    job_desc_text: raw job description text
    resumes_df: DataFrame with columns ['candidate_name', 'resume_text']
    required_skills: checklist to score against; when omitted it is
        extracted from the job description itself.
    with_evidence: also compute the per-domain breakdown and the matched-span
        payload the review interface draws. Off by default: it scans every
        resume once per required skill and keeps a copy of the resume text,
        which is worth paying for the handful of candidates a reviewer is
        looking at, and not worth paying across a 10,000-resume benchmark.
        Scores are identical either way -- this adds reporting, not scoring.
    Returns a ranked results DataFrame.
    """
    if required_skills is None:
        required_skills = extract_skills_from_jd(job_desc_text)

    columns = [
        "Rank", "Baseline_Rank", "Candidate", "Baseline_TFIDF_%", "Hybrid_Match_%",
        "Delta", "Coverage", "Matched_Skills", "Missing_Skills", "Matched_Count",
        "Total_Skills", "Verdict", "Top_Contributing_Terms", "Reason",
    ]
    if with_evidence:
        columns = columns + ["Domain_Coverage", "Evidence"]
    if resumes_df.empty:
        return pd.DataFrame(columns=columns)

    # ---- Baseline: plain TF-IDF (keyword-only) ----
    corpus_plain = [job_desc_text] + resumes_df["resume_text"].fillna("").tolist()
    tfidf_plain = TfidfVectorizer(stop_words="english")
    matrix_plain = tfidf_plain.fit_transform(corpus_plain)
    baseline_scores = cosine_similarity(matrix_plain[0:1], matrix_plain[1:]).flatten()

    # ---- Hybrid: synonym-normalized TF-IDF ----
    corpus_hybrid = [normalize_text(job_desc_text)] + [
        normalize_text(t) for t in resumes_df["resume_text"].fillna("")
    ]
    tfidf_hybrid = TfidfVectorizer(stop_words="english")
    matrix_hybrid = tfidf_hybrid.fit_transform(corpus_hybrid)
    hybrid_scores = cosine_similarity(matrix_hybrid[0:1], matrix_hybrid[1:]).flatten()

    # ---- Explainability: exact per-term contribution to the hybrid score ----
    # TfidfVectorizer L2-normalizes each row by default, so cosine similarity
    # between the JD vector and a resume vector equals their dot product:
    # sum over terms of (jd_weight_i * resume_weight_i). That means we can
    # decompose the score exactly, term by term, with no approximation —
    # this is the "Feature Importance" step, adapted for a similarity-based
    # model instead of a trained classifier. Each term also carries its share
    # of the total score, which is what the contribution bars in the UI plot.
    feature_names = tfidf_hybrid.get_feature_names_out()
    jd_vector = matrix_hybrid[0].toarray().flatten()

    results = []
    for i, row in resumes_df.reset_index(drop=True).iterrows():
        matched, missing = extract_matched_skills(row["resume_text"], required_skills)
        coverage = len(matched) / len(required_skills) if required_skills else 0

        resume_vector = matrix_hybrid[i + 1].toarray().flatten()
        contributions = jd_vector * resume_vector
        score = float(contributions.sum())
        top_idx = contributions.argsort()[::-1][:8]
        top_terms = [
            {
                "term": feature_names[idx].replace("_", " "),
                "value": round(float(contributions[idx]), 4),
                "share": round(float(contributions[idx]) / score * 100, 1) if score > 0 else 0.0,
            }
            for idx in top_idx if contributions[idx] > 0
        ]

        baseline = round(float(baseline_scores[i]) * 100, 1)
        hybrid = round(float(hybrid_scores[i]) * 100, 1)

        record = {
            "Candidate": row["candidate_name"],
            "Baseline_TFIDF_%": baseline,
            "Hybrid_Match_%": hybrid,
            "Delta": round(hybrid - baseline, 1),
            "Coverage": round(coverage, 4),
            "Matched_Skills": matched,
            "Missing_Skills": missing,
            "Matched_Count": len(matched),
            "Total_Skills": len(required_skills),
            "Verdict": verdict_for_coverage(coverage),
            "Top_Contributing_Terms": top_terms,
        }
        if with_evidence:
            # Per-domain coverage drives the radar chart; the evidence payload
            # drives the in-resume highlighter. Both are presentation of data
            # the scoring already produced -- neither changes any score.
            record["Domain_Coverage"] = domain_coverage(matched, required_skills)
            record["Evidence"] = skill_evidence(row["resume_text"], required_skills)
        record["Reason"] = justification(record)
        results.append(record)

    results_df = pd.DataFrame(results)

    # Rank under the baseline scoring too, so the interface can show how the
    # synonym-normalisation step reorders the shortlist.
    baseline_order = results_df["Baseline_TFIDF_%"].rank(
        ascending=False, method="first").astype(int)
    results_df["Baseline_Rank"] = baseline_order

    results_df = results_df.sort_values("Hybrid_Match_%", ascending=False)
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
    return results_df[columns]
