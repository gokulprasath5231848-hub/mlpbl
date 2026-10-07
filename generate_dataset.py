"""
Synthetic resume dataset generator.

Everything this produces is GENERATED, not collected: no real resume, no real
person and no real company appears anywhere in the output. That is a
deliberate choice, not a shortcut. Because the generator decides which skills
go into each resume, it also knows the correct answer for every row, and that
ground truth is what makes a quantitative evaluation possible at all — the
skill matcher can be scored against what was actually written, and the ranking
can be scored against who is actually qualified.

Two properties matter for this project specifically:

  1. Every skill mention is written either in its abbreviated form ("ML",
     "NLP", "AWS") or spelled out in full ("Machine Learning", "Natural
     Language Processing", "Amazon Web Services"). The generator records which
     style each resume used, so the synonym-normalisation step can be measured
     on exactly the resumes it is supposed to help.

  2. Resumes carry skills from outside their own role family as noise, so the
     matching problem is not trivially separable.

Outputs (written to dataset/):
  resumes_10k.csv        candidate_name, resume_text  -- app-compatible
  ground_truth.csv       candidate_name, role, seniority, phrasing, true_skills
  job_descriptions/      one .txt per role family
  relevance.csv          job_id, candidate_name, relevant (0/1), true_coverage

Usage:  python generate_dataset.py [--n 10000] [--seed 42]
"""
import argparse
import csv
import os
import random

# ---------------------------------------------------------------------
# Skill surface forms. The first entry is the spelled-out form, the second
# the abbreviation. Where a skill has no natural abbreviation the two are
# the same, which is realistic and keeps the phrasing split honest.
# ---------------------------------------------------------------------
SURFACE = {
    "machine learning":            ("Machine Learning", "ML"),
    "natural language processing": ("Natural Language Processing", "NLP"),
    "artificial intelligence":     ("Artificial Intelligence", "AI"),
    "deep learning":               ("Deep Learning", "DL"),
    "computer vision":             ("Computer Vision", "OpenCV"),
    "sql":                         ("Structured Query Language", "SQL"),
    "cloud":                       ("Amazon Web Services", "AWS"),
    "kubernetes":                  ("Kubernetes", "K8s"),
    "ci/cd":                       ("Continuous Integration", "CI/CD"),
    "javascript":                  ("JavaScript", "JS"),
    "react":                       ("React.js", "React"),
    "node.js":                     ("Node.js", "NodeJS"),
    "rest api":                     ("RESTful APIs", "REST APIs"),
    "postgresql":                  ("PostgreSQL", "Postgres"),
    "mongodb":                     ("MongoDB", "Mongo"),
    "power bi":                    ("Power BI", "PowerBI"),
    "scikit-learn":                ("scikit-learn", "sklearn"),
    "python":                      ("Python", "Python"),
    "java":                        ("Java", "Java"),
    "spring boot":                 ("Spring Boot", "SpringBoot"),
    "tensorflow":                  ("TensorFlow", "TensorFlow"),
    "pytorch":                     ("PyTorch", "PyTorch"),
    "pandas":                      ("pandas", "pandas"),
    "numpy":                       ("NumPy", "NumPy"),
    "statistics":                  ("Statistics", "Statistics"),
    "data analysis":               ("Data Analysis", "Data Analytics"),
    "text classification":         ("Text Classification", "Text Classification"),
    "tableau":                     ("Tableau", "Tableau"),
    "excel":                       ("Excel", "Excel"),
    "docker":                      ("Docker", "Docker"),
    "git":                         ("Git", "Git"),
    "linux":                       ("Linux", "Linux"),
    "html/css":                    ("HTML", "CSS"),
}

# ---------------------------------------------------------------------
# Role families: the skills that define the role, and the ones that
# commonly appear alongside without defining it.
# ---------------------------------------------------------------------
ROLES = {
    "ml_engineer": {
        "title": "Machine Learning Engineer",
        "core": ["python", "machine learning", "scikit-learn", "deep learning",
                 "tensorflow", "pytorch"],
        "optional": ["natural language processing", "computer vision", "numpy",
                     "pandas", "cloud", "docker", "sql", "statistics"],
    },
    "data_scientist": {
        "title": "Data Scientist",
        "core": ["python", "machine learning", "statistics", "pandas",
                 "data analysis", "sql"],
        "optional": ["scikit-learn", "numpy", "deep learning", "tableau",
                     "power bi", "text classification", "cloud"],
    },
    "nlp_engineer": {
        "title": "NLP Engineer",
        "core": ["python", "natural language processing", "text classification",
                 "deep learning", "pytorch"],
        "optional": ["machine learning", "scikit-learn", "artificial intelligence",
                     "numpy", "cloud", "docker"],
    },
    "data_analyst": {
        "title": "Data Analyst",
        "core": ["sql", "excel", "data analysis", "tableau", "power bi"],
        "optional": ["python", "pandas", "statistics", "postgresql"],
    },
    "backend_developer": {
        "title": "Backend Developer",
        "core": ["java", "spring boot", "rest api", "sql", "postgresql"],
        "optional": ["docker", "git", "linux", "mongodb", "kubernetes", "python"],
    },
    "frontend_developer": {
        "title": "Frontend Developer",
        "core": ["javascript", "react", "html/css", "git"],
        "optional": ["node.js", "rest api", "docker"],
    },
    "devops_engineer": {
        "title": "DevOps Engineer",
        "core": ["docker", "kubernetes", "ci/cd", "linux", "cloud"],
        "optional": ["python", "git", "postgresql", "java"],
    },
    "data_engineer": {
        "title": "Data Engineer",
        "core": ["python", "sql", "postgresql", "cloud", "docker"],
        "optional": ["kubernetes", "pandas", "mongodb", "linux", "ci/cd"],
    },
}

SENIORITY = [
    ("Intern",  0, 1, 0.55),
    ("Junior",  1, 3, 0.70),
    ("Mid",     3, 6, 0.85),
    ("Senior",  6, 12, 1.00),
]

FIRST = """Aarav Aditi Akash Ananya Aniket Anjali Arjun Bhavna Chetan Darshan
Deepa Divya Farhan Gaurav Harini Ishaan Jaya Kabir Kavya Lakshmi Manish Meera
Mohit Naveen Neha Nikhil Nisha Pallavi Pooja Pranav Priya Rahul Rajesh Rakesh
Ramya Ravi Rhea Rohan Sahana Sanjay Shreya Siddharth Sneha Sonal Suresh Swathi
Tanvi Tarun Uma Varun Vidya Vikram Vishal Yamini Zoya Abhay Bhargav Charu
Dhruv Esha Ganesh Hema Indira Jatin Kiran Lalita Madhav Nandini Omkar Parvati
Qamar Ritu Sameer Trisha Udhav Vinay Yash""".split()

LAST = """Agarwal Balan Chandran Desai Deshpande Gupta Iyer Jain Joshi Kapoor
Krishnan Kulkarni Kumar Menon Mehta Nair Naidu Pillai Prasad Raghavan Rao Reddy
Sharma Shetty Singh Subramanian Varma Venkatesh Verma Bhatt Chopra Dubey Ghosh
Hegde Kaur Malhotra Nambiar Patel Saxena Tiwari""".split()

DEGREES = ["B.E. Computer Science", "B.Tech Information Technology",
           "B.Sc Statistics", "M.Tech Computer Science", "M.Sc Data Science",
           "B.Tech Electronics", "MCA", "B.E. Information Science"]

# Sentence templates. {s} is filled with a rendered skill phrase.
OPENERS = [
    "{sen} {title} with {yrs} {yw} of experience.",
    "{title} ({sen}) — {yrs} {yw} building production systems.",
    "{sen}-level {title}. {yrs} {yw} of hands-on delivery experience.",
    "{title} with {yrs} {yw} across product and platform teams.",
]
SKILL_SENTENCES = [
    "Strong working knowledge of {s}.",
    "Day-to-day experience with {s}.",
    "Delivered production work using {s}.",
    "Comfortable with {s}.",
    "Built and maintained systems based on {s}.",
    "Hands-on with {s}.",
    "Used {s} extensively across several projects.",
    "Proficient in {s}.",
    "Applied {s} to solve real business problems.",
    "Skilled in {s}.",
]
# Split by seniority: an intern has not led a team, and saying so would make
# the generated text obviously wrong to anyone reading a sample.
PROJECTS_JUNIOR = [
    "Contributed to the {area} pipeline under senior review.",
    "Fixed defects in the {area} service and added regression tests.",
    "Built an internal tool that cut {area} turnaround by {pct}%.",
    "Presented project work at the team's fortnightly demo.",
    "Wrote documentation for the {area} module.",
]
PROJECTS_SENIOR = [
    "Led a team of {n} engineers on an internal platform rewrite.",
    "Owned the {area} pipeline end to end, from ingestion to reporting.",
    "Reduced {area} processing time by {pct}% through profiling and caching.",
    "Mentored {n} junior engineers and ran the weekly design review.",
    "Shipped {n} major releases in the last year with no critical incidents.",
    "Collaborated with product and design on the {area} roadmap.",
]
AREAS = ["reporting", "recommendation", "search", "billing", "onboarding",
         "analytics", "ingestion", "fraud-detection", "personalisation"]
CLOSERS = [
    "{degree}, graduated {year}.",
    "{degree} ({year}).",
    "Education: {degree}, {year}.",
]


def render(skill, style, rng):
    """Render a skill in the requested phrasing style."""
    full, abbr = SURFACE[skill]
    if style == "full":
        return full
    if style == "abbr":
        return abbr
    return rng.choice((full, abbr))          # mixed


def join_skills(skills, style, rng):
    parts = [render(s, style, rng) for s in skills]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def make_resume(rng, role_key):
    role = ROLES[role_key]
    sen_name, lo, hi, depth = rng.choice(SENIORITY)
    yrs = rng.randint(lo, hi) or 1

    # phrasing style for this candidate -- recorded, so the synonym step can
    # be measured on the resumes it is meant to help
    style = rng.choices(["full", "abbr", "mixed"], weights=[0.3, 0.3, 0.4])[0]

    # --- choose the skills actually possessed (this is the ground truth) ---
    n_core = max(2, round(len(role["core"]) * depth))
    skills = set(rng.sample(role["core"], min(n_core, len(role["core"]))))

    n_opt = rng.randint(0, max(1, round(len(role["optional"]) * depth * 0.6)))
    if n_opt:
        skills |= set(rng.sample(role["optional"], min(n_opt, len(role["optional"]))))

    # noise: a couple of skills from an unrelated role family
    if rng.random() < 0.45:
        other = rng.choice([k for k in ROLES if k != role_key])
        pool = ROLES[other]["core"] + ROLES[other]["optional"]
        skills |= set(rng.sample(pool, rng.randint(1, 2)))

    skills = sorted(skills)

    # --- build the text ---
    lines = [rng.choice(OPENERS).format(
        sen=sen_name, title=role["title"], yrs=yrs,
        yw="year" if yrs == 1 else "years")]

    shuffled = skills[:]
    rng.shuffle(shuffled)
    i = 0
    while i < len(shuffled):
        chunk = shuffled[i:i + rng.randint(1, 3)]
        lines.append(rng.choice(SKILL_SENTENCES).format(
            s=join_skills(chunk, style, rng)))
        i += len(chunk)

    projects = PROJECTS_JUNIOR if sen_name in ("Intern", "Junior") else PROJECTS_SENIOR
    for _ in range(rng.randint(1, 3)):
        lines.append(rng.choice(projects).format(
            n=rng.randint(2, 9), area=rng.choice(AREAS),
            pct=rng.choice([15, 20, 25, 30, 35, 40, 45, 50, 60])))

    lines.append(rng.choice(CLOSERS).format(
        degree=rng.choice(DEGREES), year=rng.randint(2012, 2025)))

    return " ".join(lines), skills, sen_name, style


def make_job_description(role_key, rng):
    role = ROLES[role_key]
    required = role["core"] + rng.sample(role["optional"],
                                         min(2, len(role["optional"])))
    # job descriptions lean on abbreviations, which is the realistic case and
    # the one plain keyword matching handles worst
    body = (
        f"Job Title: {role['title']}\n\n"
        f"We are hiring a {role['title']} to join a growing engineering team.\n"
        f"The successful candidate will have strong experience with "
        f"{join_skills(required[:4], 'abbr', rng)}.\n"
        f"Additional experience with {join_skills(required[4:], 'abbr', rng)} "
        f"is preferred.\n"
        f"You will work closely with product and design, own features end to "
        f"end, and contribute to technical decisions across the team.\n"
    )
    return body, sorted(set(required))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "job_descriptions"), exist_ok=True)

    role_keys = list(ROLES)
    rows, truth = [], []
    used_names = set()

    for i in range(args.n):
        role_key = role_keys[i % len(role_keys)]
        text, skills, sen, style = make_resume(rng, role_key)

        # unique synthetic name
        while True:
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if name not in used_names:
                used_names.add(name)
                break
            if len(used_names) > len(FIRST) * len(LAST) * 0.6:
                name = f"{name} {len(used_names)}"
                used_names.add(name)
                break

        rows.append({"candidate_name": name, "resume_text": text})
        truth.append({"candidate_name": name, "role": role_key,
                      "seniority": sen, "phrasing": style,
                      "true_skills": "|".join(skills)})

    with open(os.path.join(args.out, "resumes_10k.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_name", "resume_text"])
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.out, "ground_truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_name", "role", "seniority",
                                          "phrasing", "true_skills"])
        w.writeheader()
        w.writerows(truth)

    # --- job descriptions + relevance judgements -------------------------
    truth_by_name = {t["candidate_name"]: set(t["true_skills"].split("|"))
                     for t in truth}
    rel_rows = []
    for role_key in role_keys:
        body, required = make_job_description(role_key, rng)
        with open(os.path.join(args.out, "job_descriptions",
                               f"{role_key}.txt"), "w") as f:
            f.write(body)

        # A candidate is relevant to a job when their GROUND-TRUTH skills
        # cover at least 60% of what the job asks for. This judgement never
        # looks at the resume text, so it cannot favour either scoring method.
        for name, have in truth_by_name.items():
            cov = len(have & set(required)) / len(required)
            rel_rows.append({"job_id": role_key, "candidate_name": name,
                             "relevant": int(cov >= 0.6),
                             "true_coverage": round(cov, 3)})

    with open(os.path.join(args.out, "relevance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "candidate_name",
                                          "relevant", "true_coverage"])
        w.writeheader()
        w.writerows(rel_rows)

    n_rel = sum(r["relevant"] for r in rel_rows)
    print(f"Wrote {len(rows)} resumes to {args.out}/resumes_10k.csv")
    print(f"Ground truth        : {args.out}/ground_truth.csv")
    print(f"Job descriptions    : {len(role_keys)}")
    print(f"Relevance judgements: {len(rel_rows)} rows, {n_rel} positive "
          f"({n_rel / len(rel_rows) * 100:.1f}%)")
    styles = {}
    for t in truth:
        styles[t["phrasing"]] = styles.get(t["phrasing"], 0) + 1
    print(f"Phrasing split      : {styles}")


if __name__ == "__main__":
    main()
