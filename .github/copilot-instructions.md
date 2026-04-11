# Information Retrieval Project Guidelines

## Specific Tasks Completed in this Repository
This project acts as an end-to-end opinion search engine focused on "AI in Education". The core tasks completed and maintained in this repository are:
- **Data Crawling & Consolidation**: Automated scraping pipelines across 5 platforms (Quora, YouTube, X/Twitter, LinkedIn, Reddit) yielding ~70,000 records, alongside scripts for formatting, relevance filtering, and deduplication.
- **Sentiment Classification**: Implementation of sentiment analysis (VADER/custom) to categorize opinions, complete with evaluation scripts against ground-truth datasets.
- **Data Indexing**: A fully configured Apache Solr backend (`solr/configset/`) with custom schemas, synonyms, and stopwords for efficient information retrieval.
- **Web-Based Opinion Search Engine**: A Flask-based UI (`app/app.py`) enabling rich querying, multifaceted filtering (by source and sentiment), timeline visualizations, and dynamically generated word clouds/charts.

## Component Boundaries
- **Data Crawling & Preparation**: Scraper logic lives in `data_scrapping_scripts/`. Formatting, filtering, and corpus consolidation tools live in `tools/` (e.g. `consolidate_corpus.py`, `check_relevance.py`). Output is consolidated into `data/analysis/master_corpus.csv`. 
- **Search Engine UI**: Built with Flask in `app/app.py` with templates in `app/templates/`. Handles querying the Apache Solr instance and visualizing multifaceted search logic.
- **Classification**: Sentiment analysis models run via `classify.py`. Accuracy metrics and tests are checked against an evaluation set via `evaluate.py`.
- **Indexing**: Apache Solr configurations live in `solr/configset/`. Orchestrated by Docker using `docker-compose.yml`.

## Build and Test
- **Start Solr Backend**: 
  ```bash
  docker compose up -d
  ```
- **Rebuild Master Corpus**: 
  ```bash
  python tools/consolidate_corpus.py
  ```
- **Generate Predictions & Evaluate Classification**:
  ```bash
  python classify.py --eval-only
  python evaluate.py
  ```
- **Run the Web Interface**: 
  ```bash
  python app/app.py
  ```

## Conventions
- **Data handling**: When adding or updating data ingestion tools, ensure unicode normalization, whitespace collapsing, and preservation of data provenance fields (like `source`, `url`).
- **Data visualization**: When adding Matplotlib visualizations to `app.py`, generate the plot, render to a bytes buffer, and return base64 encoded strings to cleanly inject charts directly into the HTML templates without writing temporary files to disk.
- **Dependencies**: Follow existing library standards (e.g. `pandas` for tabular data formats, `pysolr` for interacting with the Solr backend).

See [README.md](README.md) for detailed descriptions of the queries, data sources, and the full text cleaning pipeline.