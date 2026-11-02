"""Web UI for DocuMind (Days 11–12).

A single dependency-free HTML page served by the REST API server at
``GET /`` (and ``GET /index.html``). It has four jobs:

* **Upload** — a form posting a document (optional id/title plus text)
  to ``POST /ingest``, showing the outcome (added / updated /
  unchanged / duplicate) and chunk count.
* **Ask** — a form posting a question to ``POST /qa``, showing the
  extractive answer and its structured citations.
* **Sources panel** (Day 12) — a live list of the indexed documents
  fetched from ``GET /documents`` (title, id, chunk count), refreshed
  on page load and after every successful upload; clicking a source
  fetches ``GET /documents/<doc_id>`` and shows its stored text as a
  preview.
* **History** (Day 12) — past questions kept in the browser's
  ``localStorage`` (newest first, capped at 20), each clickable to
  restore its answer and citations, with a button to clear the lot.

No server-side templating: the page is static HTML with inline CSS and
a small amount of vanilla JavaScript that talks to the JSON API with
``fetch``. User-supplied text is rendered with ``textContent`` (never
as markup), so a document or answer can never inject markup into
the page — that rule covers the sources panel and history too, since
both render stored document titles, questions and answers.
"""

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocuMind — Ask your documents</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px;
         margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
  p.sub { color: #555; margin-top: 0; }
  section { border: 1px solid #ddd; border-radius: 8px;
            padding: 1rem 1.25rem; margin: 1.25rem 0; }
  label { display: block; font-weight: 600; margin: 0.75rem 0 0.25rem; }
  input[type=text], textarea, select {
    width: 100%; box-sizing: border-box; padding: 0.5rem;
    border: 1px solid #bbb; border-radius: 4px; font: inherit; }
  textarea { min-height: 8rem; }
  button { margin-top: 0.75rem; padding: 0.5rem 1.25rem;
           font: inherit; cursor: pointer; }
  button.linklike { background: none; border: none; padding: 0;
                    color: #0b5394; text-decoration: underline;
                    text-align: left; }
  .result { margin-top: 1rem; padding: 0.75rem; background: #f6f6f6;
            border-radius: 4px; white-space: pre-wrap; }
  .result.error { background: #fdecea; }
  .citation { margin: 0.5rem 0; padding-left: 0.75rem;
              border-left: 3px solid #888; }
  .citation b { display: block; }
  .citation span { color: #555; font-size: 0.9rem; }
  #status { color: #555; }
  #sources-list, #history-list { padding-left: 1.25rem; }
  #sources-list li, #history-list li { margin: 0.4rem 0; }
  #source-preview { max-height: 16rem; overflow: auto; }
  .empty { color: #555; font-style: italic; }
</style>
</head>
<body>
<h1>DocuMind</h1>
<p class="sub">Ask questions over your own documents, with citations.
Pure Python, no external services.</p>
<p id="status">Loading index status…</p>

<section>
<h2>Upload a document</h2>
<form id="upload-form">
  <label for="upload-doc-id">Document id (optional — defaults to the title)</label>
  <input type="text" id="upload-doc-id" placeholder="e.g. my-notes.md">
  <label for="upload-title">Title (optional)</label>
  <input type="text" id="upload-title" placeholder="e.g. My notes">
  <label for="upload-text">Text</label>
  <textarea id="upload-text" required
    placeholder="Paste the document text here…"></textarea>
  <button type="submit">Upload</button>
</form>
<div id="upload-result" class="result" hidden></div>
</section>

<section>
<h2>Ask a question</h2>
<form id="ask-form">
  <label for="ask-question">Question</label>
  <input type="text" id="ask-question" required
    placeholder="e.g. What is AEM built on?">
  <label for="ask-method">Ranking method</label>
  <select id="ask-method">
    <option value="tfidf" selected>TF-IDF</option>
    <option value="bm25">BM25</option>
    <option value="combined">Combined</option>
  </select>
  <button type="submit">Ask</button>
</form>
<div id="ask-answer" class="result" hidden></div>
<div id="ask-citations" hidden></div>
</section>

<section id="sources-panel">
<h2>Indexed sources</h2>
<p class="sub">Every document currently in the index. Click one to
preview its stored text.</p>
<button type="button" id="refresh-sources">Refresh sources</button>
<ul id="sources-list"></ul>
<p id="sources-empty" class="empty" hidden>No documents indexed yet.</p>
<div id="source-preview" class="result" hidden></div>
</section>

<section id="history-panel">
<h2>Question history</h2>
<p class="sub">Past questions, kept in this browser only. Click one to
see its answer and citations again.</p>
<button type="button" id="clear-history">Clear history</button>
<ul id="history-list"></ul>
<p id="history-empty" class="empty" hidden>No questions asked yet.</p>
</section>

<script>
async function postJSON(path, payload) {
  const resp = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const body = await resp.json();
  if (!resp.ok) { throw new Error(body.error || ("HTTP " + resp.status)); }
  return body;
}

function show(el, text, isError) {
  el.hidden = false;
  el.textContent = text;
  el.classList.toggle("error", !!isError);
}

async function refreshStatus() {
  const el = document.getElementById("status");
  try {
    const resp = await fetch("/health");
    const body = await resp.json();
    el.textContent = body.documents + " document(s), " +
      body.chunks + " chunk(s) indexed (DocuMind " + body.version + ").";
  } catch (e) {
    el.textContent = "Could not reach the API: " + e.message;
  }
}

function renderCitations(citesEl, body) {
  citesEl.hidden = false;
  citesEl.textContent = "";
  const heading = document.createElement("h3");
  heading.textContent = "Sources (" + body.source_count + ")";
  citesEl.appendChild(heading);
  body.citations.forEach((c) => {
    const div = document.createElement("div");
    div.className = "citation";
    const label = document.createElement("b");
    label.textContent = c.citation + (c.title ? " — " + c.title : "");
    const snippet = document.createElement("div");
    snippet.textContent = c.snippet;
    const scores = document.createElement("span");
    scores.textContent = "score " + c.score.toFixed(3) +
      (c.rerank_score !== undefined ?
        ", rerank " + c.rerank_score.toFixed(3) : "");
    div.appendChild(label);
    div.appendChild(snippet);
    div.appendChild(scores);
    citesEl.appendChild(div);
  });
}

async function refreshSources() {
  const list = document.getElementById("sources-list");
  const empty = document.getElementById("sources-empty");
  const preview = document.getElementById("source-preview");
  try {
    const resp = await fetch("/documents");
    const body = await resp.json();
    if (!resp.ok) { throw new Error(body.error || ("HTTP " + resp.status)); }
    list.textContent = "";
    empty.hidden = body.documents.length > 0;
    body.documents.forEach((doc) => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "linklike";
      btn.textContent = (doc.title || doc.doc_id) + " — " +
        doc.doc_id + " (" + doc.num_chunks + " chunk(s))";
      btn.addEventListener("click", async () => {
        try {
          const detail = await fetch("/documents/" +
            encodeURIComponent(doc.doc_id));
          const detailBody = await detail.json();
          if (!detail.ok) {
            throw new Error(detailBody.error || ("HTTP " + detail.status));
          }
          show(preview, detailBody.content);
        } catch (err) {
          show(preview, "Could not load source: " + err.message, true);
        }
      });
      li.appendChild(btn);
      list.appendChild(li);
    });
  } catch (e) {
    list.textContent = "";
    empty.hidden = true;
    show(preview, "Could not load sources: " + e.message, true);
  }
}

const HISTORY_KEY = "documind-history";
const HISTORY_LIMIT = 20;

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function saveHistory(entry) {
  const history = loadHistory();
  history.unshift(entry);
  const trimmed = history.slice(0, HISTORY_LIMIT);
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  } catch (e) {
    // Private browsing or a full store: history is best-effort only.
  }
  return trimmed;
}

function renderHistory() {
  const history = loadHistory();
  const list = document.getElementById("history-list");
  const empty = document.getElementById("history-empty");
  list.textContent = "";
  empty.hidden = history.length > 0;
  history.forEach((entry) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "linklike";
    btn.textContent = entry.question;
    btn.addEventListener("click", () => {
      document.getElementById("ask-question").value = entry.question;
      if (entry.method) {
        document.getElementById("ask-method").value = entry.method;
      }
      show(document.getElementById("ask-answer"), entry.answer);
      renderCitations(document.getElementById("ask-citations"), entry);
    });
    li.appendChild(btn);
    list.appendChild(li);
  });
}

document.getElementById("refresh-sources").addEventListener("click", () => {
  refreshSources();
});

document.getElementById("clear-history").addEventListener("click", () => {
  try { localStorage.removeItem(HISTORY_KEY); } catch (e) {}
  renderHistory();
});

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const out = document.getElementById("upload-result");
  const payload = {
    text: document.getElementById("upload-text").value
  };
  const docId = document.getElementById("upload-doc-id").value.trim();
  const title = document.getElementById("upload-title").value.trim();
  if (docId) { payload.doc_id = docId; }
  if (title) { payload.title = title; }
  try {
    const body = await postJSON("/ingest", payload);
    show(out, "Outcome: " + body.outcome + " — " + body.doc_id +
      " (" + body.num_chunks + " chunk(s)).");
    refreshStatus();
    refreshSources();
  } catch (err) {
    show(out, "Upload failed: " + err.message, true);
  }
});

document.getElementById("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const answerEl = document.getElementById("ask-answer");
  const citesEl = document.getElementById("ask-citations");
  try {
    const question = document.getElementById("ask-question").value;
    const method = document.getElementById("ask-method").value;
    const body = await postJSON("/qa", {
      question: question,
      method: method
    });
    show(answerEl, body.answer);
    renderCitations(citesEl, body);
    saveHistory({question: question, method: method,
                 answer: body.answer, citations: body.citations,
                 source_count: body.source_count});
    renderHistory();
  } catch (err) {
    show(answerEl, "Ask failed: " + err.message, true);
    citesEl.hidden = true;
  }
});

refreshStatus();
refreshSources();
renderHistory();
</script>
</body>
</html>
"""
