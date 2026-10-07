# Demo script

Everything below uses only the files in this folder. Start the app with
`python app.py` and open http://127.0.0.1:5000

---

## 1. The basic run (2 minutes)

1. Press **Load sample** — this fills the Machine Learning Engineer job
   description.
2. Watch the **Required skills** chips appear as the text lands. Point out that
   these were read from the description, not hard-coded, and that they can be
   edited or removed.
3. Upload `sample_resumes_large.csv` (12 candidates) and press **Run screening**.
4. On the review screen: the ranked rail on the left, coverage meters, verdict
   badges. Click any candidate.

**What to say:** every candidate is scored twice, and the checklist verdict is a
separate signal from the similarity score.

---

## 2. The synonym effect — the core claim (2 minutes)

This is the part worth rehearsing. Use the file-upload path, not the CSV.

1. Start a **New screening**, press **Load sample** again for the ML job
   description.
2. Under **Individual files**, upload all three:
   - `Nandini_Rao.pdf` — spells every skill out ("Machine Learning",
     "Natural Language Processing", "Amazon Web Services")
   - `Arun_Bhatt.docx` — the same skills, abbreviated ("ML", "NLP", "AWS")
   - `Karan_Mehta.pdf` — a backend developer, as a control
3. Run it, then toggle **Baseline ⇄ Hybrid** at the top of the rail.

**What happens:** under Baseline, Nandini ranks above Arun (34.1% vs 28.4%).
Under Hybrid, the order flips — Arun 36.1%, Nandini 34.7% — and the gap closes
from 5.7 points to 1.4. Both show identical 5/9 skill coverage the whole time,
which is the giveaway that they were always equivalent candidates.

**What to say:** plain keyword matching ranked one candidate above an equally
qualified one purely because of how they wrote their skills. That gap is the
bug this project exists to fix, and the coverage meter proves the two were
equivalent all along.

This run also demonstrates PDF and DOCX parsing, and that uploaded files and
CSV rows can be combined in one ranked list.

---

## 3. The checklist is not hard-coded (1 minute)

1. **New screening**, then paste in `backend_job_description.txt`.
2. Point at the skill chips: they are now `java`, `spring boot`, `rest api`,
   `sql`, `postgresql`, `docker`, `git`, `kubernetes`, `linux` — a completely
   different list.
3. Upload the same three resume files and run.

**What happens:** Karan Mehta scores 9/9 and a STRONG MATCH; the two ML
candidates drop to 1/9 and WEAK. Same tool, same code, different role.

---

## 4. If you have time

- **Pool insights** tab — which requirements are scarcest across the pool.
  Better with the 12-row CSV than with three files.
- **Compare** tab — pick two or three candidates side by side.
- **Threshold slider** — drag it and watch every verdict re-label while the
  scores stay fixed. Useful for explaining that the verdict is a policy choice
  and the score is a measurement.
- **Contribution bars** on a mid-ranked candidate — often a generic word like
  "experience" is the largest single contributor. Worth showing honestly: it is
  what the explainability layer is for.
- **Export CSV** — includes the written reason for every candidate.

---

## Files in this folder

| File | What it is |
|---|---|
| `sample_job_description.txt` | Machine Learning Engineer role |
| `backend_job_description.txt` | Backend Developer role, for step 3 |
| `sample_resumes.csv` | 3 candidates, the original small set |
| `sample_resumes_large.csv` | 12 candidates, best for pool insights and compare |
| `Nandini_Rao.pdf` | ML candidate, skills spelled out in full |
| `Arun_Bhatt.docx` | Same skills, abbreviated |
| `Karan_Mehta.pdf` | Backend candidate, control for step 3 |

The CSVs need exactly two columns: `candidate_name` and `resume_text`.
