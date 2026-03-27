"""
Reddit Data Fetcher
====================
Fetches posts and comments from Reddit using:
  1. The public JSON API (no auth needed, read-only)
  2. Optionally, the official API via PRAW (if installed + credentials provided)

Usage:
    python reddit_scraper.py                         # Auto-fetch AI in Education data
    python reddit_scraper.py -s python -l 10         # Top 10 from r/python
    python reddit_scraper.py -s python -c hot -l 5   # Hot 5 from r/python
    python reddit_scraper.py -s python --comments     # Include top comments
    python reddit_scraper.py -s python --save csv     # Save to CSV
    python reddit_scraper.py -s python --save json    # Save to JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote

# ── Configuration ────────────────────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
BASE_URL = "https://www.reddit.com"
REQUEST_DELAY = 5  # seconds between requests; need adjust abit if it doesnt work on ur sys

# ── AI in Education — Subreddits & Search Queries ────────────────────────────

AI_EDUCATION_SUBREDDITS = [
    "artificial",
    "education",
    "edtech",
    "ChatGPT",
    "MachineLearning",
    "learnmachinelearning",
    "highereducation",
    "Teachers",
    "college",
    "computerscience",
    "OnlineEducation",
    "elearning",
]

AI_EDUCATION_SEARCH_QUERIES = [
    "AI in education",
    "artificial intelligence education",
    "AI tutoring",
    "ChatGPT classroom",
    "AI learning tools",
    "machine learning education",
    "AI academic integrity",
    "AI personalized learning",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crawled_data")

# ── Helpers ──────────────────────────────────────────────────────────────────


def fetch_json(url: str, _retries: int = 0) -> dict | None:
    """Fetch JSON from a URL using curl (bypasses TLS fingerprinting blocks)."""
    max_retries = 3
    try:
        result = subprocess.run(
            ["curl", "-s", "-w", "\n%{http_code}", "--max-time", "15",
             "-H", f"User-Agent: {USER_AGENT}",
             url],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            print(f"⚠  curl error (exit {result.returncode}) for: {url}")
            return None
        # Split body and status code
        parts = result.stdout.rsplit("\n", 1)
        body = parts[0] if len(parts) == 2 else result.stdout
        status = int(parts[1]) if len(parts) == 2 else 0
        if status == 429:
            if _retries >= max_retries:
                print(f"⚠  Rate limited {max_retries} times, skipping: {url}")
                return None
            wait = 10 * (_retries + 1)
            print(f"⚠  Rate limited by Reddit. Waiting {wait} seconds... (retry {_retries + 1}/{max_retries})")
            time.sleep(wait)
            return fetch_json(url, _retries + 1)
        if status >= 400:
            print(f"⚠  HTTP {status} for: {url}")
            return None
        return json.loads(body)
    except subprocess.TimeoutExpired:
        print(f"⚠  Request timed out: {url}")
        return None
    except json.JSONDecodeError:
        print(f"⚠  Invalid JSON from: {url}")
        return None


def ts_to_str(ts: float) -> str:
    """Convert a Unix timestamp to a readable datetime string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def truncate(text: str, length: int = 200) -> str:
    """Truncate text to a given length."""
    if not text:
        return ""
    return text[:length] + ("…" if len(text) > length else "")


# ── Core Functions ───────────────────────────────────────────────────────────


def get_subreddit_posts(
    subreddit: str,
    category: str = "hot",
    limit: int = 10,
    time_filter: str = "week",
) -> list[dict]:
    """
    Fetch posts from a subreddit.

    Args:
        subreddit:   Name of the subreddit (without r/)
        category:    One of: hot, new, top, rising
        limit:       Number of posts to fetch (max ~100 per request)
        time_filter: For 'top' category — hour, day, week, month, year, all
    """
    url = f"{BASE_URL}/r/{subreddit}/{category}.json?limit={limit}&t={time_filter}"
    data = fetch_json(url)
    if not data:
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        p = child["data"]
        posts.append(
            {
                "id": p["id"],
                "title": p["title"],
                "author": p.get("author", "[deleted]"),
                "score": p["score"],
                "upvote_ratio": p.get("upvote_ratio", 0),
                "num_comments": p["num_comments"],
                "created_utc": ts_to_str(p["created_utc"]),
                "url": p.get("url", ""),
                "permalink": f"https://reddit.com{p['permalink']}",
                "selftext": p.get("selftext", ""),
                "is_self": p.get("is_self", False),
                "flair": p.get("link_flair_text", ""),
                "subreddit": p.get("subreddit", subreddit),
            }
        )
    return posts


def get_post_comments(permalink: str, limit: int = 10) -> list[dict]:
    """
    Fetch top-level comments for a given post.

    Args:
        permalink: The post's permalink path (e.g., /r/python/comments/abc123/...)
        limit:     Number of comments to return
    """
    # Strip domain if full URL was passed
    permalink = permalink.replace("https://reddit.com", "").replace("https://www.reddit.com", "")
    url = f"{BASE_URL}{permalink}.json?limit={limit}&sort=top"
    data = fetch_json(url)

    comments = []
    if not data or len(data) < 2:
        return comments

    for child in data[1].get("data", {}).get("children", []):
        if child["kind"] != "t1":
            continue
        c = child["data"]
        comments.append(
            {
                "id": c["id"],
                "author": c.get("author", "[deleted]"),
                "score": c.get("score", 0),
                "body": c.get("body", ""),
                "created_utc": ts_to_str(c["created_utc"]),
            }
        )
    return comments


def search_reddit(query: str, subreddit: str | None = None, limit: int = 10) -> list[dict]:
    """
    Search Reddit for posts matching a query.

    Args:
        query:     Search terms
        subreddit: Optionally restrict search to a subreddit
        limit:     Number of results
    """
    encoded_query = quote(query)
    if subreddit:
        url = f"{BASE_URL}/r/{subreddit}/search.json?q={encoded_query}&restrict_sr=on&limit={limit}"
    else:
        url = f"{BASE_URL}/search.json?q={encoded_query}&limit={limit}"
    data = fetch_json(url)
    if not data:
        return []

    return [
        {
            "title": p["data"]["title"],
            "subreddit": p["data"]["subreddit"],
            "score": p["data"]["score"],
            "permalink": f"https://reddit.com{p['data']['permalink']}",
            "author": p["data"].get("author", "[deleted]"),
        }
        for p in data.get("data", {}).get("children", [])
    ]


# ── Output / Export ──────────────────────────────────────────────────────────


def print_posts(posts: list[dict], show_text: bool = False) -> None:
    """Pretty-print a list of posts to the terminal."""
    for i, p in enumerate(posts, 1):
        print(f"\n{'─' * 70}")
        print(f"  #{i}  {p['title']}")
        print(f"  ↑ {p['score']:>6}  | 💬 {p['num_comments']}  | by u/{p['author']}  | {p['created_utc']}")
        if p.get("flair"):
            print(f"  🏷  {p['flair']}")
        print(f"  🔗 {p['permalink']}")
        if show_text and p.get("selftext"):
            print(f"  📝 {truncate(p['selftext'], 300)}")
    print(f"\n{'─' * 70}")


def print_comments(comments: list[dict]) -> None:
    """Pretty-print a list of comments."""
    for c in comments:
        print(f"  ↑ {c['score']:>5}  u/{c['author']}  ({c['created_utc']})")
        print(f"         {truncate(c['body'], 200)}")
        print()


def save_to_csv(posts: list[dict], filename: str) -> None:
    """Save posts to a CSV file."""
    if not posts:
        print("No data to save.")
        return
    keys = posts[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(posts)
    print(f"✅ Saved {len(posts)} posts to {filename}")


def save_to_json(posts: list[dict], filename: str) -> None:
    """Save posts to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(posts)} posts to {filename}")


# ── Auto-fetch: AI in Education ──────────────────────────────────────────────


def auto_fetch_ai_education(
    posts_per_subreddit: int = 25,
    posts_per_query: int = 25,
    include_comments: bool = True,
    comments_per_post: int = 5,
) -> list[dict]:
    """
    Automatically fetch Reddit data related to "AI in Education".

    Output format matches the other crawled_data CSVs:
        source, url, question_title, answer_text, scraped_at

    Each post's selftext becomes one row, and each comment becomes its own
    row (same pattern Quora uses for multiple answers per question).

    Steps:
        1. Scrapes relevant subreddits (hot + top/month).
        2. Runs search queries across Reddit.
        3. Deduplicates by post ID.
        4. Optionally fetches top comments per post.
        5. Saves everything to crawled_data/redditcrawl.csv.
    """
    seen_ids: set[str] = set()
    all_posts: list[dict] = []

    def _collect(posts: list[dict]) -> None:
        for p in posts:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                all_posts.append(p)

    # ── 1. Subreddit browsing ────────────────────────────────────────────
    for sub in AI_EDUCATION_SUBREDDITS:
        for category, time_filter in [("hot", "week"), ("top", "month")]:
            print(f"  Fetching r/{sub}/{category} (t={time_filter}) ...")
            posts = get_subreddit_posts(sub, category, posts_per_subreddit, time_filter)
            _collect(posts)
            time.sleep(REQUEST_DELAY)

    # ── 2. Search queries ────────────────────────────────────────────────
    for query in AI_EDUCATION_SEARCH_QUERIES:
        print(f"  Searching: \"{query}\" ...")
        results = search_reddit(query, limit=posts_per_query)
        for r in results:
            if r.get("permalink"):
                post_id = r["permalink"].rstrip("/").split("/")[-2] if "/comments/" in r["permalink"] else ""
                if post_id and post_id not in seen_ids:
                    seen_ids.add(post_id)
                    all_posts.append({
                        "id": post_id,
                        "title": r.get("title", ""),
                        "permalink": r["permalink"],
                        "selftext": "",
                        "subreddit": r.get("subreddit", ""),
                    })
        time.sleep(REQUEST_DELAY)

    print(f"\n  Collected {len(all_posts)} unique posts.")

    # ── 3. Build rows in the shared schema ───────────────────────────────
    # Format: source, url, question_title, answer_text, scraped_at
    scraped_at = datetime.now(tz=timezone.utc).isoformat()
    rows: list[dict] = []

    for i, p in enumerate(all_posts):
        title = p.get("title", "")
        url = p.get("permalink", "")

        # Row for the post's own selftext (if any)
        selftext = p.get("selftext", "").strip()
        if selftext:
            rows.append({
                "source": "reddit",
                "url": url,
                "question_title": title,
                "answer_text": selftext,
                "scraped_at": scraped_at,
            })

        # Rows for each comment
        if include_comments:
            comments = get_post_comments(url, limit=comments_per_post)
            for c in comments:
                body = c.get("body", "").strip()
                if body:
                    rows.append({
                        "source": "reddit",
                        "url": url,
                        "question_title": title,
                        "answer_text": body,
                        "scraped_at": scraped_at,
                    })
            if (i + 1) % 20 == 0:
                print(f"    ... processed {i + 1}/{len(all_posts)} posts")
            time.sleep(REQUEST_DELAY)

    # ── 4. Save to CSV ───────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "redditcrawl.csv")

    fieldnames = ["source", "url", "question_title", "answer_text", "scraped_at"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Saved {len(rows)} rows ({len(all_posts)} posts) to {out_path}")
    return rows


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch posts and comments from Reddit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reddit_scraper.py -s python -l 10
  python reddit_scraper.py -s python -c top -t month -l 25
  python reddit_scraper.py -s python --comments --save csv
  python reddit_scraper.py --search "machine learning" -l 5
        """,
    )
    parser.add_argument("-s", "--subreddit", help="Subreddit to fetch (without r/)")
    parser.add_argument(
        "-c", "--category", default="hot", choices=["hot", "new", "top", "rising"],
        help="Post category (default: hot)",
    )
    parser.add_argument("-l", "--limit", type=int, default=10, help="Number of posts (default: 10)")
    parser.add_argument(
        "-t", "--time", default="week", choices=["hour", "day", "week", "month", "year", "all"],
        help="Time filter for 'top' category (default: week)",
    )
    parser.add_argument("--comments", action="store_true", help="Also fetch top comments per post")
    parser.add_argument("--search", help="Search Reddit for a query instead of browsing")
    parser.add_argument("--save", choices=["csv", "json"], help="Save results to a file")
    parser.add_argument("--show-text", action="store_true", help="Show self-text preview for posts")
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-fetch AI in Education data from predefined subreddits & queries",
    )
    parser.add_argument(
        "--no-comments", action="store_true",
        help="Skip fetching comments in auto mode (faster)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # If no arguments provided or --auto flag, run auto AI-in-Education fetch
    if len(sys.argv) == 1 or args.auto:
        print("=" * 60)
        print("  Reddit Data Fetcher — AI in Education (Auto Mode)")
        print("=" * 60)
        print(f"\n  Subreddits: {', '.join(AI_EDUCATION_SUBREDDITS)}")
        print(f"  Search queries: {len(AI_EDUCATION_SEARCH_QUERIES)}")
        print(f"  Comments: {'off' if args.no_comments else 'on'}\n")
        auto_fetch_ai_education(include_comments=not args.no_comments)
        return

    # Search mode
    if args.search:
        print(f"Searching Reddit for: '{args.search}'...")
        results = search_reddit(args.search, args.subreddit, args.limit)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['subreddit']}] {r['title']}  (↑{r['score']})")
            print(f"     {r['permalink']}")
        return

    # Subreddit browsing mode
    if not args.subreddit:
        parser.error("Please provide --subreddit or --search")

    print(f"Fetching {args.limit} {args.category} posts from r/{args.subreddit}...")
    posts = get_subreddit_posts(args.subreddit, args.category, args.limit, args.time)

    if not posts:
        print("No posts found.")
        return

    print_posts(posts, show_text=args.show_text)

    # Optionally fetch comments
    if args.comments:
        for p in posts:
            print(f"\n💬 Comments for: {truncate(p['title'], 60)}")
            time.sleep(REQUEST_DELAY)
            comments = get_post_comments(p["permalink"])
            p["comments"] = comments
            if comments:
                print_comments(comments)

    # Save if requested
    if args.save:
        filename = f"reddit_{args.subreddit}_{args.category}.{args.save}"
        (save_to_csv if args.save == "csv" else save_to_json)(posts, filename)


if __name__ == "__main__":
    main()