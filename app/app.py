"""
Flask web UI for the Opinions search engine backed by Apache Solr.

Innovations:
  - Multifaceted search (source + sentiment facets)
  - Timeline search (date range filtering)
  - Enhanced search (sentiment pie chart, source bar chart, word cloud)
"""

import base64
import io
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pysolr
from flask import Flask, render_template, request
from wordcloud import WordCloud

app = Flask(__name__)
SOLR_URL = "http://localhost:8983/solr/opinions"
solr = pysolr.Solr(SOLR_URL, timeout=30)

RESULTS_PER_PAGE = 10


def build_chart_base64(fig):
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_wordcloud(texts):
    """Word cloud from result texts."""
    combined = " ".join(texts)
    if not combined.strip():
        return None

    wc = WordCloud(
        width=1600,
        height=600,
        background_color="white",
        max_words=80,
        colormap="viridis",
        scale=2,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return build_chart_base64(fig)


def parse_facet_pairs(facet_list):
    """Convert Solr's flat facet list [name, count, name, count, ...] to a dict."""
    result = {}
    for i in range(0, len(facet_list), 2):
        result[facet_list[i]] = facet_list[i + 1]
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return render_template("index.html", error="Please enter a search query.")

    # Collect filter parameters
    sources = request.args.getlist("source")
    sentiment = request.args.get("sentiment", "")
    emotion = request.args.get("emotion", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    page = max(1, int(request.args.get("page", 1)))

    # Build Solr query params
    fq_list = []
    if sources:
        source_filter = " OR ".join(f'source:"{s}"' for s in sources)
        fq_list.append(f"({source_filter})")
    if sentiment:
        fq_list.append(f'sentiment:"{sentiment}"')
    if emotion:
        fq_list.append(f'emotion:"{emotion}"')
    if date_from or date_to:
        d_from = date_from + "T00:00:00Z" if date_from else "*"
        d_to = date_to + "T23:59:59Z" if date_to else "*"
        fq_list.append(f"date:[{d_from} TO {d_to}]")

    start = (page - 1) * RESULTS_PER_PAGE

    kwargs = {
        "q": q,
        "defType": "edismax",
        "qf": "title^0.5 text^4",
        "pf": "text^8",
        "mm": "2<75%",
        "start": start,
        "rows": RESULTS_PER_PAGE,
        "fl": "id,source,url,title,text,date,sentiment,sentiment_score,subjectivity,subjectivity_score,emotion,emotion_score",
        "fq": fq_list,
        "hl": "true",
        "hl.fl": "text",
        "hl.snippets": 1,              # 🔥 ONLY ONE snippet
        "hl.fragsize": 300,            # 🔥 bigger, cleaner context
        "hl.mergeContiguous": "true",  # 🔥 prevents overlap duplication
        "hl.method": "unified",        # 🔥 better snippet quality
        "hl.simple.pre": "<mark>",
        "hl.simple.post": "</mark>",
    }

    # Execute search with timing
    t0 = time.time()
    results = solr.search(**kwargs)

    # Extract Solr QTime (IMPORTANT)
    solr_qtime = results.raw_response["responseHeader"]["QTime"]

    query_time = round((time.time() - t0) * 1000, 2)

    total_hits = results.hits
    total_pages = max(1, (total_hits + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)

    # Parse highlighting
    highlighting = results.highlighting if hasattr(results, "highlighting") else {}

    # Parse facets
    facet_fields = results.facets.get("facet_fields", {}) if hasattr(results, "facets") else {}
    source_facets = parse_facet_pairs(facet_fields.get("source", []))
    sentiment_facets = parse_facet_pairs(facet_fields.get("sentiment", []))
    emotion_facets = parse_facet_pairs(facet_fields.get("emotion", []))

    # Build results list
    docs = []
    for doc in results:
        doc_id = doc.get("id", "")
        hl = highlighting.get(doc_id, {})
        snippet = ""

        if hl.get("text"):
            snippet = hl["text"][0]
        else:
            text = doc.get("text", "")
            snippet = text[:300] + "..." if len(text) > 300 else text

        docs.append({
            "id": doc_id,
            "source": doc.get("source", ""),
            "url": doc.get("url", ""),
            "title": doc.get("title", "No Title"),
            "snippet": snippet,
            "date": doc.get("date", ""),
            "sentiment": doc.get("sentiment", ""),
            "sentiment_score": doc.get("sentiment_score", 0),
            "subjectivity": doc.get("subjectivity", ""),
            "subjectivity_score": doc.get("subjectivity_score", 0),
            "emotion": doc.get("emotion", ""),
            "emotion_score": doc.get("emotion_score", 0),
        })

    # Word cloud from current page texts
    texts = [doc.get("text", "") for doc in results]
    wordcloud_img = generate_wordcloud(texts) if texts else None

    return render_template(
        "results.html",
        query=q,
        docs=docs,
        total_hits=total_hits,
        query_time=query_time,
        solr_qtime=solr_qtime,
        page=page,
        total_pages=total_pages,
        source_facets=source_facets,
        sentiment_facets=sentiment_facets,
        emotion_facets=emotion_facets,
        active_sources=sources,
        active_sentiment=sentiment,
        active_emotion=emotion,
        date_from=date_from,
        date_to=date_to,
        wordcloud_img=wordcloud_img,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
