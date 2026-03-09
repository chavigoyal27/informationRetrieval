import re
import asyncio
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse, urljoin

from playwright.async_api import async_playwright
from ftfy import fix_text
import unicodedata


# configs
TARGET_RECORDS = 30000
TARGET_URLS = 5000
MAX_ANSWERS_PER_URL = 50

MIN_CHARS = 250

SCROLL_ROUNDS = 5
SCROLL_PAUSE = 0.25
DELAY_BETWEEN_PAGES = 1.0

OUTPUT_FILE = "quora_data.csv"


# helpers
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def looks_gated(body_lower: str) -> bool:
    return ("captcha" in body_lower) or ("unusual traffic" in body_lower) or (
        "sign up" in body_lower and "log in" in body_lower
    )

def clean_unicode(text):
    text = fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    return text

BAD_SUBSTRINGS = [
    "Sponsored by",
    "Promoted by",
    "Advertisement",
    "Related questions",
    "More answers below",
    "Skip to content",
    "Skip to search",
    "Sign In",
    "Sign Up",
    "Log in",
    "© Quora",
]

AUTHOR_BIO_RE = re.compile(r"(Author has \d|answer views|Originally Answered)", re.IGNORECASE)


def is_noise(t: str) -> bool:

    t_lower = t.lower()

    # filters
    if any(b.lower() in t_lower for b in BAD_SUBSTRINGS):
        return True

    if AUTHOR_BIO_RE.search(t):
        return True

    if t.count("Upvote") >= 2:
        return True

    # filters for Quora navigation blocks
    if "/unanswered/" in t_lower:
        return True

    if "contributors" in t_lower:
        return True

    if "logical and psychological thinking" in t_lower:
        return True

    # discard corrupted multi-language encoding blocks
    if "‡" in t:
        return True

    return False


# url expansion
async def expand_quora_urls_async(seed_urls, target=150, headless=False, delay=1.0):
    def looks_like_question(u: str) -> bool:
        p = urlparse(u)
        host = (p.hostname or "").lower()

        # allow ONLY main Quora (blocks *.quora.com spaces)
        if host not in ("www.quora.com", "quora.com"):
            return False
        if p.query or p.fragment:
            return False
        if not p.path or p.path == "/":
            return False

        bad_prefixes = ("/profile/", "/topic/", "/answer/", "/signin", "/login", "/about", "/terms", "/privacy")
        if p.path.lower().startswith(bad_prefixes):
            return False
        return True

    seen = set(seed_urls)
    queue = list(seed_urls)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )

        page = await context.new_page()

        while queue and len(seen) < target:
            url = queue.pop(0)
            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                status = resp.status if resp else None

                # quick debug: are we being gated?
                body = (await page.inner_text("body")).lower()
                gated = ("captcha" in body) or ("sign up" in body and "log in" in body)

                print(f"Visited: {url[:55]}... status={status} gated={gated} seen={len(seen)}")

                if gated:
                    await asyncio.sleep(delay)
                    continue

                # collect hrefs (relative + absolute)
                hrefs = await page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => e.getAttribute('href'))"
                )

                added = 0
                for h in hrefs:
                    if not h:
                        continue
                    abs_u = urljoin("https://www.quora.com", h)
                    if abs_u.endswith("/"):
                        abs_u = abs_u[:-1]
                    if looks_like_question(abs_u) and abs_u not in seen:
                        seen.add(abs_u)
                        queue.append(abs_u)
                        added += 1
                        if len(seen) >= target:
                            break

                print(f"  +{added} new urls (total={len(seen)})")

            except Exception as e:
                print("Error visiting:", url, "|", str(e)[:120])

            await asyncio.sleep(delay)

        await context.close()
        await browser.close()

    return list(seen)


# scraper
async def scrape_quora(urls, out_csv="quora_answers.csv"):

    rows = []
    seen = set()

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            viewport={"width":1280,"height":900}
        )

        page = await context.new_page()

        for i, url in enumerate(urls, start=1):

            if len(rows) >= TARGET_RECORDS:
                break

            try:

                print(f"\nVisiting: {url}")

                await page.goto(url, wait_until="networkidle", timeout=30000)

                await page.wait_for_timeout(2000)

                # remove login modal
                try:
                    await page.evaluate("""
                    () => {
                        const modal = document.querySelector('[role="dialog"]')
                        if (modal) modal.remove()
                    }
                    """)
                except:
                    pass


                # scroll to load answers
                for _ in range(SCROLL_ROUNDS):
                    await page.mouse.wheel(0,2500)
                    await page.wait_for_timeout(int(SCROLL_PAUSE*1000))


                # click Continue Reading buttons
                try:
                    await page.evaluate("""
                    () => {
                        const els = Array.from(document.querySelectorAll('a,button,div[role="button"]'))
                        for (const el of els) {
                            const t = (el.innerText || '').toLowerCase().trim()
                            if (t === "continue reading") el.click()
                        }
                    }
                    """)
                except:
                    pass

                await page.wait_for_timeout(5000)


                # extract answers
                answers = await page.evaluate("""
                () => {

                    const results = []

                    const blocks = document.querySelectorAll("span.q-text, div.q-text")

                    for (const b of blocks) {

                        const text = (b.innerText || "")
                            .replace(/\\s+/g," ")
                            .trim()

                        if (text.length > 200) {
                            results.push(text)
                        }

                    }

                    return results
                }
                """)
                print("answers found:", len(answers))


                added = 0

                for t in answers:

                    t = norm(t)

                    if len(t) < MIN_CHARS:
                        continue

                    if is_noise(t):
                        continue

                    key = t[:200]

                    # checks if answer is new and unique
                    if key in seen:
                        continue

                    seen.add(key)

                    rows.append({
                        "source": "quora",
                        "url": url,
                        "question_title": "",
                        "answer_text": t,
                        "scraped_at": datetime.utcnow().isoformat() + "Z"
                    })

                    added += 1

                    if added >= MAX_ANSWERS_PER_URL:
                        break

                    if len(rows) >= TARGET_RECORDS:
                        break


                print(f"[{i}/{len(urls)}] +{added} | total={len(rows)}")


            except Exception as e:

                print("Error:", str(e)[:120])

            await asyncio.sleep(DELAY_BETWEEN_PAGES)


        await browser.close()


    df = pd.DataFrame(rows)

    df["answer_text"] = df["answer_text"].apply(clean_unicode)

    # Fix remaining encoding artifacts
    df["answer_text"] = df["answer_text"].str.replace("‚Ä¢", "•")
    df["answer_text"] = df["answer_text"].str.replace("‚Äî", "—")
    df["answer_text"] = df["answer_text"].str.replace("‚Ä¶", "…")

    df["answer_text"] = df["answer_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    
    df = df.drop_duplicates(subset=["answer_text"])

    print("Final dataset size after dedup:", len(df))

    df.to_csv(out_csv,index=False,encoding="utf-8")

    return df


async def main():

    seed = [
        "https://www.quora.com/What-is-your-opinion-on-using-artificial-intelligence-AI-for-teaching-and-learning-Do-you-believe-it-will-enhance-or-diminish-the-quality-of-education",
        "https://www.quora.com/How-can-college-teachers-stop-students-from-using-AI-for-homework",
        "https://www.quora.com/What-are-the-potential-risks-of-relying-heavily-on-AI-in-education-and-how-can-they-be-mitigated",
        "https://www.quora.com/Should-students-be-allowed-to-use-ChatGPT-and-other-AI-tools-for-school-projects",
        "https://www.quora.com/Whats-your-opinion-on-AI-in-education",
        "https://www.quora.com/What-are-the-pros-and-cons-of-using-AI-in-education",
        "https://www.quora.com/What-are-the-benefits-of-artificial-intelligence-in-education-How-can-AI-help-students-learn-better-than-humans",
        "https://www.quora.com/How-can-AI-be-used-to-enhance-education-and-learning",
        "https://www.quora.com/How-are-teachers-actually-using-AI-in-their-classrooms-right-now",
        "https://www.quora.com/Is-AI-useful-for-students-and-education"
    ]

    urls = await expand_quora_urls_async(seed, target=TARGET_URLS, headless=False)

    print("Collected URLs:", len(urls))

    df = await scrape_quora(urls, OUTPUT_FILE)

    print("Finished. Rows:", len(df))


if __name__ == "__main__":
    asyncio.run(main())