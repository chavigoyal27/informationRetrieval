import re
import time
import hashlib
from tqdm import tqdm
import pandas as pd
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

# ================= ENV =================
env_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(env_path)
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing YOUTUBE_API_KEY in .env")

# ================= SETTINGS =================
TARGET_COMMENTS = 10000
MAX_VIDEOS_PER_QUERY = 30
COMMENTS_PER_VIDEO = 120
SLEEP_SEC = 0.1

# ================= QUERIES =================
SEARCH_QUERIES = [

# ================= NEGATIVE =================

# Harm / Risk
"dangers of ai in education",
"why ai is harmful in schools",
"negative effects of chatgpt",
"risks of using ai for homework",
"ai harming student learning",
"why ai is dangerous for students",
"problems caused by chatgpt in education",
"ai misuse in classrooms",
"ai ruining education",
"ai destroying learning",
"ai making students lazy",
"students relying too much on chatgpt",
"ai reducing critical thinking",
"ai lowering education standards",
"ai hurting academic skills",
"ai weakening problem solving skills",
"chatgpt is harmful for students",
"ai is bad for education",
"chatgpt is dangerous",
"ai should not be used in schools",

# Cheating / Dishonesty
"students cheating using chatgpt",
"ai plagiarism in schools",
"chatgpt cheating cases",
"academic dishonesty with ai",
"ai tools used for cheating exams",
"problems with ai generated assignments",
"turnitin ai detection issues",
"students misusing ai tools",
"chatgpt essay cheating",
"ai academic fraud cases",

# Teachers / Job Threat
"teachers hate chatgpt",
"teachers complaining about ai",
"ai replacing teachers debate negative",
"ai threatening teaching jobs",
"ai cannot replace teachers argument",
"problems with ai grading",
"teachers against ai tools",
"ai destroying teacher roles",
"why educators oppose ai",
"ai vs teachers controversy",

# Policy / Ban
"schools banning chatgpt",
"should chatgpt be banned",
"ai banned in schools debate",
"should ai be banned in education",
"arguments against ai in schools",
"ai banned in classrooms debate",
"legal issues of ai in education",
"ethical concerns of ai in schools",
"restrictions on ai tools in schools",
"controversy over ai in education",

# Ethics / Privacy
"ethical issues of ai in education",
"bias in ai education tools",
"ai fairness problems in learning",
"privacy concerns ai students",
"data misuse in ai education",
"ai exploiting student data",
"ai causing inequality in education",
"ai discrimination in learning systems",

# Question-style negative
"is ai bad for students",
"should students use chatgpt or not",
"why is ai harmful in education",
"is chatgpt cheating",
"should ai be banned in schools",


]

NUM_QUERIES = len(SEARCH_QUERIES)
PER_QUERY_TARGET = TARGET_COMMENTS // NUM_QUERIES

query_counts = {}

# ================= HELPERS =================
def normalize_text(t: str) -> str:
    t = t.replace("\n", " ").replace("\r", " ")
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
                    "parent_id": thread["snippet"]["topLevelComment"]["id"],
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

# ================= MAIN =================
records = []
seen_ids = set()
seen_text = set()

pbar = tqdm(desc="Crawling YouTube comments")

for q in SEARCH_QUERIES:

    # initialize query counter
    if q not in query_counts:
        query_counts[q] = 0

    videos = search_videos(q, max_results=MAX_VIDEOS_PER_QUERY)
    seen_videos = set()

    for v in videos:

        # 🔥 STOP if query quota reached
        if query_counts[q] >= PER_QUERY_TARGET:
            break

        vid = v["video_id"]
        if vid in seen_videos:
            continue
        seen_videos.add(vid)

        try:
            rows = fetch_comments(vid, cap=COMMENTS_PER_VIDEO)
        except Exception:
            continue

        kept = 0
        too_short = 0
        dup = 0

        for row in rows:

            # 🔥 STOP if query quota reached
            if query_counts[q] >= PER_QUERY_TARGET:
                break

            cid = row["comment_id"]
            if cid in seen_ids:
                dup += 1
                continue

            raw_text = row["text"].replace("\r", " ").replace("\n", " ")
            txt = normalize_text(raw_text)

            if len(txt.split()) < 3:
                too_short += 1
                continue

            th = text_hash(txt)
            if th in seen_text:
                dup += 1
                continue

            seen_ids.add(cid)
            seen_text.add(th)

            records.append({
                **v,
                **row,
                "text_norm": txt,
                "query": q
            })

            query_counts[q] += 1
            kept += 1
            pbar.update(1)

        time.sleep(SLEEP_SEC)

        print(f"VIDEO {vid} | raw={len(rows)} kept={kept} short={too_short} dup={dup}")

pbar.close()

# ================= SAVE =================
df = pd.DataFrame(records)
df.to_csv("corpus_youtube_ai_education.csv", index=False, encoding="utf-8", lineterminator="\n")

print("\nSaved:", len(df), "records")

# ================= STATS =================
if not df.empty:
    words = df["text_norm"].str.split().map(len).sum()
    types = len(set(" ".join(df["text_norm"].tolist()).split()))

    print("Total words:", words)
    print("Unique word types:", types)

# ================= DEBUG =================
print("\nPer-query distribution (sample):")
for k, v in list(query_counts.items())[:10]:
    print(k, ":", v)