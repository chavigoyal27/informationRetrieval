# SC4021 Information Retrieval - Opinion Search Engine

## Topic: AI in Education

An opinion search engine that crawls, indexes, and performs sentiment analysis on public opinions about artificial intelligence in education.

---

## Data Crawling

### Sources & Methods

| Source | Library | Method | Records |
|--------|---------|--------|---------|
| Quora | Playwright (async browser automation), ftfy | Expands from 10 seed questions to 5,000 unique URLs, extracts up to 50 answers per page. Unicode normalization and deduplication applied. | 30,000 |
| YouTube | YouTube Data API v3 (google-api-python-client) | Searches 32 targeted queries, collects up to 30 videos per query and 120 comments per video. Deduplicates by comment ID and text hash. | 30,000 |
| X (Twitter) | Twikit (unofficial API client) | Generates 300+ query combinations from AI terms x education terms. Uses cookie-based auth with exponential backoff for rate limits. | 6,498 |
| LinkedIn | Playwright with human-like scrolling | Searches 84 queries (hashtags, product names, keyword combos). Simulates realistic user behavior with random pauses and scroll-backs. | 2,830 |
| Reddit | curl via subprocess (public JSON API, no auth) | Browses 12 subreddits (hot/top) and runs 8 search queries. Fetches up to 5 comments per post. | 704 |

### Search Queries / Keywords

The crawlers target a broad range of AI-in-education topics:

- **General:** "AI in education", "artificial intelligence in education", "future of AI in education"
- **Generative AI:** "ChatGPT in education", "ChatGPT for students", "ChatGPT for teachers"
- **Tools & Platforms:** Khanmigo, Duolingo AI, Coursera, Google Classroom, Copilot
- **Concerns:** "AI cheating in schools", "AI plagiarism detection", "AI academic integrity"
- **Policy:** "should schools allow AI", "AI regulation in education", "AI bans in schools"
- **Future:** "AI replacing teachers", "AI transforming education", "personalized learning AI"

### Subreddits Crawled

r/artificial, r/education, r/ChatGPT, r/MachineLearning, r/learnmachinelearning, r/highereducation, r/Teachers, r/college, r/computerscience, r/OnlineEducation, r/elearning, r/edtech

### Data Storage

Raw crawled data is stored in `data/crawled/`.

| File | Columns |
|------|---------|
| `quoracrawl.csv` | source, url, question_title, answer_text, scraped_at |
| `youtubecrawl.csv` | video_id, video_title, channel_title, published_at, comment_id, parent_id, author, like_count, text, text_norm, query |
| `twitterxcrawl.csv` | source, url, question_title, answer_text |
| `linkedincrawl.csv` | source, url, question_title, answer_text |
| `redditcrawl.csv` | source, url, question_title, answer_text, scraped_at |

### Tools

| Script | Purpose |
|--------|---------|
| `tools/reddit_filter.py` | Filters Reddit data to keep only rows mentioning both AI and education keywords |
| `tools/reddit_format.py` | Cleans Reddit text (removes markdown, URLs, normalizes whitespace, drops entries <15 chars) |
| `tools/consolidate_corpus.py` | Merges all source CSVs into a single `master_corpus.csv` with a unified schema |
| `tools/check_relevance.py` | Checks each record for topical relevance (AI + education keywords), saves off-topic rows |
| `tools/check_sentiment.py` | Runs VADER sentiment analysis to show positive/negative/neutral distribution |
| `tools/balance_corpus.py` | Generates balanced corpus files by removing off-topic/positive Quora and downsampling |

### Master Corpus

All source CSVs are consolidated into `data/analysis/master_corpus.csv` by running:

```bash
python tools/consolidate_corpus.py
```

This normalises every source into a unified schema and applies cleaning:

| Column | Description |
|--------|-------------|
| `id` | Sequential unique identifier |
| `source` | Quora, YouTube, Twitter, LinkedIn, or Reddit |
| `url` | Link to original content |
| `title` | Post/question title (where available) |
| `text` | The opinion text content |
| `date` | Timestamp (where available) |

Cleaning steps:
- Whitespace collapsed and stripped across all fields
- Source names standardised (e.g. `"X (Twitter)"` → `"Twitter"`)
- Rows with empty text dropped
- Exact-duplicate texts removed (62 duplicates found across sources)

### Corpus Statistics (after consolidation)

| Metric | Value |
|--------|-------|
| Total records | 69,970 |
| Total words | 6,880,348 |
| Unique types | 251,683 |

#### Per-Source Breakdown

| Source | Records | Words | Avg Words/Record |
|--------|---------|-------|------------------|
| Quora | 30,000 | 5,597,179 | 187 |
| YouTube | 30,000 | 807,616 | 27 |
| Twitter | 6,439 | 324,942 | 50 |
| LinkedIn | 2,830 | 89,083 | 31 |
| Reddit | 701 | 67,645 | 96 |

### Data Quality Checks

#### Topical Relevance

Each record is checked for the presence of at least one AI keyword (e.g. `ai`, `chatgpt`, `llm`, `neural network`) **and** at least one education keyword (e.g. `student`, `school`, `curriculum`, `grading`). Run with:

```bash
python tools/check_relevance.py
```

| Source | Total | On-topic | Rate |
|--------|-------|----------|------|
| Quora | 30,000 | 9,156 | 30.5% |
| YouTube | 30,000 | 19,407 | 64.7% |
| Twitter | 6,439 | 4,676 | 72.6% |
| LinkedIn | 2,830 | 2,332 | 82.4% |
| Reddit | 701 | 697 | 99.4% |
| **Overall** | **69,970** | **36,268** | **51.8%** |

Off-topic rows are saved to `data/analysis/off_topic.csv` for review. Quora has the lowest relevance rate because many answers discuss AI or education in isolation rather than together.

#### Sentiment Distribution

VADER sentiment analysis is used as a quick health check to assess whether the corpus has a balanced mix of positive and negative opinions. Run with:

```bash
python tools/check_sentiment.py
```

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive | 45,595 | 65.2% |
| Negative | 14,300 | 20.4% |
| Neutral | 10,075 | 14.4% |

Per-source breakdown:

| Source | Pos% | Neg% | Neu% |
|--------|------|------|------|
| Quora | 77.3% | 20.4% | 2.4% |
| YouTube | 54.1% | 21.5% | 24.3% |
| Twitter | 60.2% | 20.2% | 19.6% |
| LinkedIn | 65.3% | 10.1% | 24.6% |
| Reddit | 63.9% | 21.0% | 15.1% |

Positive/Negative ratio: **3.19** — the dataset leans positive. Per-record sentiment scores are saved to `data/analysis/sentiment_distribution.csv`.

### Balanced Corpus

To address the positive skew, balanced versions of the corpus are generated by removing off-topic Quora records, all positive Quora records, and then downsampling remaining positive records. Run with:

```bash
python tools/balance_corpus.py
```

| File | Pos/Neg Ratio | Records | Words |
|------|---------------|---------|-------|
| `data/final_corpus/corpus_balanced_1to1.csv` | 1.00 | 28,664 | 1,108,322 |
| `data/final_corpus/corpus_balanced_1.5to1.csv` | 1.50 | 33,450 | 1,291,694 |

Both files use the same unified schema (`id, source, url, title, text, date`) and exceed the assignment minimums of 10,000 records and 100,000 words.

---

## Project Structure

```
informationRetrieval/
├── data/
│   ├── crawled/                # Raw crawled CSV datasets
│   │   ├── quoracrawl.csv
│   │   ├── youtubecrawl.csv
│   │   ├── twitterxcrawl.csv
│   │   ├── linkedincrawl.csv
│   │   └── redditcrawl.csv
│   ├── analysis/               # Intermediate analysis outputs
│   │   ├── master_corpus.csv
│   │   ├── off_topic.csv
│   │   └── sentiment_distribution.csv
│   └── final_corpus/           # Balanced corpus files for indexing
│       ├── corpus_balanced_1to1.csv
│       └── corpus_balanced_1.5to1.csv
├── data_scrapping_scripts/     # Crawling scripts
│   ├── quora_scraper.py
│   ├── youtube_crawl.py
│   ├── Xscraper.py
│   ├── linkedinscrap.py
│   └── reddit_scraper.py
├── tools/                      # Post-processing utilities
│   ├── reddit_filter.py
│   ├── reddit_format.py
│   ├── consolidate_corpus.py
│   ├── check_relevance.py
│   ├── check_sentiment.py
│   └── balance_corpus.py
└── Assignment.pdf
```
