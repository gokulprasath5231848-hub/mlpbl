# Deploying

## Why there is no separate frontend to host

This is a server-rendered Flask application. `templates/index.html` and
`templates/results.html` are Jinja templates: Flask fills them in on each
request with `results_json`, `insights`, `strong_threshold` and
`moderate_threshold`. Netlify, GitHub Pages and Vercel's static hosting serve
files as they are on disk — they cannot run Python, so those templates would
reach the browser as literal `{{ }}` placeholders.

So the app deploys as **one service on Render**. Netlify is not involved.

## Deploy to Render

1. Push the project to GitHub (`.gitignore` here already excludes
   `__pycache__/` and virtualenvs).
2. Render dashboard → **New → Web Service** → connect the repository.
3. If Render picks up `render.yaml`, accept it. Otherwise set by hand:
   - Runtime: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --threads 4 --timeout 120`
   - Environment variable: `PYTHON_VERSION` = `3.11.9`
4. Deploy. Render supplies `$PORT`; gunicorn binds it automatically.

## Things that will bite you

**Keep `--workers 1`.** `app.py` holds the last run in a module-level global
(`_last_results_df`) so `/download` can re-serve it. With two workers the
download request can land on a process that never ran the screening and will
return an empty file.

**Free instances sleep when idle.** The first request after a quiet period
waits for the container to start again, which can take the better part of a
minute. Before a live demo, open the URL once and let it wake up — or
screen-record the demo as a fallback.

**`dataset/` is about 8 MB.** It is only needed by `evaluate_large.py`, which
never runs on the server. Leaving it in the repo just makes each deploy
slower; removing it from the deployed branch is safe.

**Uploads are held in memory and results are not persisted.** A restart loses
the current screening. That is fine for a review session and is already stated
as a limitation in the report.

## Running it locally instead

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000
