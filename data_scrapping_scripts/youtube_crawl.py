import re
import time
import hashlib
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(env_path)
API_KEY = os.getenv("YOUTUBE_API_KEY")
print("API key loaded?", bool(API_KEY))
if not API_KEY:
    raise RuntimeError("Missing YOUTUBE_API_KEY in .env")

SEARCH_QUERIES = [

# General AI in education
"ai in education",
"artificial intelligence in education",
"future of ai in education",
"impact of ai on education",
"ai changing education",

# ChatGPT and generative AI
"chatgpt in education",
"chatgpt for students",
"chatgpt for teachers",
"chatgpt homework help",
"chatgpt school use",
"should students use chatgpt",

# Academic integrity / cheating
"ai cheating in school",
"chatgpt cheating school",
"ai plagiarism students",
"turnitin ai detector",
"detecting ai generated essays",

# AI tutoring and learning tools
"ai tutor for students",
"ai learning assistant",
"ai personalized learning",
"ai homework helper",
"ai tools for studying",

# Teachers and grading
"ai grading essays",
"ai helping teachers",
"ai lesson planning teachers",
"ai feedback for students",

# Benefits of AI
"benefits of ai in education",
"how ai helps students learn",
"ai improving education",
"ai tools for education",

# Risks / criticisms
"dangers of ai in education",
"ai harming education",
"ai replacing teachers",
"problems with ai in schools",

# Policy / regulation
"ban chatgpt in schools",
"should ai be allowed in schools",
"schools banning ai tools",
"regulation of ai in education",

# Universities and research
"ai in universities",
"ai tools for college students",
"ai research in education",

# Future of education
"future classroom with ai",
"will ai replace teachers",
"ai education technology future"
]

TARGET_COMMENTS = 30000
MAX_VIDEOS_PER_QUERY = 30  
COMMENTS_PER_VIDEO = 120
SLEEP_SEC = 0.1

def normalize_text(t: str) -> str:
    t = t.replace("\n", " ")      
    t = t.replace("\r", " ")
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def text_hash(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

youtube = build("youtube", "v3", developerKey=API_KEY)

def search_videos(query, max_results=25):
    req = youtube.search().list(
        q=query,
        part="id,snippet",
        type="video",
        maxResults=max_results,
        relevanceLanguage="en"
    )
    res = req.execute()
    vids = []
    for item in res.get("items", []):
        vids.append({
            "video_id": item["id"]["videoId"],
            "video_title": item["snippet"]["title"],
            "channel_title": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"]
        })
    return vids

def fetch_comments(video_id, cap=120):
    all_rows = []
    next_page = None

    while len(all_rows) < cap:
        req = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page,
            textFormat="plainText"
        )
        res = req.execute()

        for thread in res.get("items", []):
            if len(all_rows) >= cap:
                break

            top = thread["snippet"]["topLevelComment"]["snippet"]
            all_rows.append({
                "comment_id": thread["snippet"]["topLevelComment"]["id"],
                "parent_id": "",
                "video_id": video_id,
                "author": top.get("authorDisplayName", ""),
                "published_at": top.get("publishedAt", ""),
                "like_count": top.get("likeCount", 0),
                "text": top.get("textDisplay", "").replace("\n", " ").replace("\r", " ")
            })

            replies = thread.get("replies", {}).get("comments", [])
            for r in replies:
                if len(all_rows) >= cap:
                    break
                rs = r["snippet"]
                all_rows.append({
                    "comment_id": r["id"],
                    "parent_id":thread["snippet"]["topLevelComment"]["id"],
                    "video_id": video_id,
                    "author": rs.get("authorDisplayName", ""),
                    "published_at": rs.get("publishedAt", ""),
                    "like_count": rs.get("likeCount", 0),
                    "text": rs.get("textDisplay", "").replace("\n", " ").replace("\r", " ")
                })

        next_page = res.get("nextPageToken")
        if not next_page:
            break

        time.sleep(SLEEP_SEC)

    return all_rows

records = []
seen_ids = set()
seen_text = set()

pbar = tqdm(total=TARGET_COMMENTS, desc="Crawling YouTube comments")

for q in SEARCH_QUERIES:
    videos = search_videos(q, max_results=MAX_VIDEOS_PER_QUERY)
    seen_videos = set()

    for v in videos:
        vid = v["video_id"]
        if vid in seen_videos:
            continue
        seen_videos.add(vid)
        if len(records) >= TARGET_COMMENTS:
            break

        try:
            rows = fetch_comments(v["video_id"], cap=COMMENTS_PER_VIDEO)
        except Exception as e:
            continue
        kept = 0
        too_short = 0
        dup = 0

        for row in rows:
            if len(records) >= TARGET_COMMENTS:
                break

            cid = row["comment_id"]
            if cid in seen_ids:
                dup += 1
                continue

            raw_text = row["text"].replace("\r", " ").replace("\n", " ")
            txt = normalize_text(raw_text)
            tokens = txt.split()
            if len(tokens) < 3:
                too_short += 1
                continue

            th = text_hash(txt)
            if th in seen_text:
                dup +=1
                continue

            seen_ids.add(cid)
            seen_text.add(th)

            kept +=1

            records.append({
                **v,
                **row,
                "text_norm": txt,
                "query": q
            })
            pbar.update(1)

        time.sleep(SLEEP_SEC)
        print(
        f"VIDEO {v['video_id']} | raw={len(rows)} kept={kept} short={too_short} dup={dup}"
        )

pbar.close()

df = pd.DataFrame(records)
df.to_csv("corpus_youtube_ai_education.csv", index=False, encoding="utf-8", lineterminator="\n")
print("Saved:", len(df), "records -> corpus_youtube_ai_education.csv")

# Basic stats for your report
words = df["text_norm"].str.split().map(len).sum()
types = len(set(" ".join(df["text_norm"].tolist()).split()))
print("Total words:", words)
print("Unique word types:", types)