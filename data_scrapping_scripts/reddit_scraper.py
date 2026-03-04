"""
Reddit Data Fetcher
====================
Fetches posts and comments from Reddit using:
  1. The public JSON API (no auth needed, read-only)
  2. Optionally, the official API via PRAW (if installed + credentials provided)

Usage:
    python reddit_scraper.py                         # Interactive mode
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
REQUEST_DELAY = 1.5  # seconds between requests; need adjust abit if it doesnt work on ur sys 

# ── Helpers ──────────────────────────────────────────────────────────────────


def fetch_json(url: str) -> dict | None:
    """Fetch JSON from a URL using curl (bypasses TLS fingerprinting blocks)."""
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
            print("⚠  Rate limited by Reddit. Waiting 10 seconds...")
            time.sleep(10)
            return fetch_json(url)
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
    return parser


def interactive_mode():
    """Run in interactive mode when no arguments are provided."""
    print("=" * 50)
    print("  Reddit Data Fetcher — Interactive Mode")
    print("=" * 50)

    subreddit = input("\nSubreddit (e.g. python): ").strip().lstrip("r/")
    if not subreddit:
        print("No subreddit provided. Exiting.")
        return

    category = input("Category [hot/new/top/rising] (default: hot): ").strip() or "hot"
    limit = int(input("Number of posts (default: 10): ").strip() or "10")
    fetch_comments = input("Fetch comments too? [y/N]: ").strip().lower() == "y"

    print(f"\nFetching {limit} {category} posts from r/{subreddit}...")
    posts = get_subreddit_posts(subreddit, category, limit)

    if not posts:
        print("No posts found.")
        return

    print_posts(posts, show_text=True)

    if fetch_comments:
        for p in posts:
            print(f"\n💬 Comments for: {truncate(p['title'], 60)}")
            time.sleep(REQUEST_DELAY)
            comments = get_post_comments(p["permalink"])
            p["comments"] = comments
            if comments:
                print_comments(comments)
            else:
                print("  (no comments)")

    save = input("\nSave results? [csv/json/N]: ").strip().lower()
    if save in ("csv", "json"):
        filename = f"reddit_{subreddit}_{category}.{save}"
        (save_to_csv if save == "csv" else save_to_json)(posts, filename)


def main():
    parser = build_parser()
    args = parser.parse_args()

    # If no arguments provided, run interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
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