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
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_sentiment_chart(facet_counts):
    """Pie chart of sentiment distribution from facet counts."""
    labels = []
    sizes = []
    colors_map = {
        "positive": "#4CAF50",
        "negative": "#F44336",
        "neutral": "#FFC107",
    }
    colors = []
    for label, count in facet_counts.items():
        if count > 0:
            labels.append(f"{label} ({count})")
            sizes.append(count)
            colors.append(colors_map.get(label, "#999999"))

    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Sentiment Distribution")
    return build_chart_base64(fig)


def generate_source_chart(facet_counts):
    """Bar chart of source distribution from facet counts."""
    labels = []
    sizes = []
    for label, count in sorted(facet_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            labels.append(label)
            sizes.append(count)

    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(5, 3))
    bar_colors = ["#2196F3", "#FF9800", "#9C27B0", "#009688", "#795548"]
    ax.barh(labels, sizes, color=bar_colors[: len(labels)])
    ax.set_xlabel("Number of Results")
    ax.set_title("Source Distribution")
    ax.invert_yaxis()
    return build_chart_base64(fig)


def generate_wordcloud(texts):
    """Word cloud from result texts."""
    combined = " ".join(texts)
    if not combined.strip():
        return None

    wc = WordCloud(
        width=600,
        height=300,
        background_color="white",
        max_words=80,
        colormap="viridis",
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud")
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
    if date_from or date_to:
        d_from = date_from + "T00:00:00Z" if date_from else "*"
        d_to = date_to + "T23:59:59Z" if date_to else "*"
        fq_list.append(f"date:[{d_from} TO {d_to}]")

    start = (page - 1) * RESULTS_PER_PAGE

    kwargs = {
        "q": q,
        "start": start,
        "rows": RESULTS_PER_PAGE,
        "fl": "id,source,url,title,text,date,sentiment,sentiment_score",
        "fq": fq_list,
        "hl": "true",
        "hl.fl": "text,title",
        "hl.snippets": 2,
        "hl.fragsize": 200,
        "hl.simple.pre": "<mark>",
        "hl.simple.post": "</mark>",
        "facet": "true",
        "facet.field": ["source", "sentiment"],
        "facet.mincount": 1,
    }

    # Execute search with timing
    t0 = time.time()
    results = solr.search(**kwargs)
    query_time = round((time.time() - t0) * 1000, 2)

    total_hits = results.hits
    total_pages = max(1, (total_hits + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)

    # Parse highlighting
    highlighting = results.highlighting if hasattr(results, "highlighting") else {}

    # Parse facets
    facet_fields = results.facets.get("facet_fields", {}) if hasattr(results, "facets") else {}
    source_facets = parse_facet_pairs(facet_fields.get("source", []))
    sentiment_facets = parse_facet_pairs(facet_fields.get("sentiment", []))

    # Build results list
    docs = []
    for doc in results:
        doc_id = doc.get("id", "")
        hl = highlighting.get(doc_id, {})
        snippet = ""
        if "text" in hl:
            snippet = " ... ".join(hl["text"])
        elif "title" in hl:
            snippet = " ... ".join(hl["title"])
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
        })

    # Generate charts from facet data (not just current page)
    sentiment_chart = generate_sentiment_chart(sentiment_facets)
    source_chart = generate_source_chart(source_facets)

    # Word cloud from current page texts
    texts = [doc.get("text", "") for doc in results]
    wordcloud_img = generate_wordcloud(texts) if texts else None

    return render_template(
        "results.html",
        query=q,
        docs=docs,
        total_hits=total_hits,
        query_time=query_time,
        page=page,
        total_pages=total_pages,
        source_facets=source_facets,
        sentiment_facets=sentiment_facets,
        active_sources=sources,
        active_sentiment=sentiment,
        date_from=date_from,
        date_to=date_to,
        sentiment_chart=sentiment_chart,
        source_chart=source_chart,
        wordcloud_img=wordcloud_img,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
