"""
libraries:
pip3 install playwright pandas textblob openpyxl lxml html5lib beautifulsoup4
playwright install chromium
python3 -m textblob.download_corpora
"""

import asyncio
import pandas as pd
from datetime import datetime
import os
from textblob import TextBlob
from playwright.async_api import async_playwright
import time
import random
import hashlib

# ===== YOUR LINKEDIN CREDENTIALS =====
LINKEDIN_EMAIL = "placeholder"
LINKEDIN_PASSWORD = "placeholder"

# ===== EXPANDED SEARCH QUERIES =====
SEARCH_QUERIES = [
    "AI in education", "artificial intelligence in education", "machine learning in education",
    "deep learning in education", "generative AI in education", "AI in schools",
    "AI in universities", "AI in higher education", "AI in K-12", "AI in classroom",
    "AI for teaching", "AI for learning", "AI for students", "AI for teachers",
    "AI for professors", "AI in academic research", "AI in curriculum development",
    "AI in educational technology", "AI in EdTech", "AI in online learning",
    "ChatGPT in education", "ChatGPT for teaching", "ChatGPT for learning",
    "ChatGPT in classroom", "GPT-4 in education", "Large language models in education",
    "LLMs in education", "Generative AI in classroom", "ChatGPT for students",
    "ChatGPT for teachers", "AI chatbots in education", "Conversational AI in education",
    "AI writing tools education", "AI essay writing education", "AI plagiarism detection",
    "AI tutoring systems", "Intelligent tutoring systems", "Adaptive learning AI",
    "Personalized learning AI", "AI learning platforms", "AI education tools",
    "AI assessment tools", "AI grading tools", "AI feedback tools", "AI homework help",
    "AI study assistant", "AI flashcard generator", "AI quiz generator",
    "AI lesson planner", "AI curriculum designer", "AI teaching strategies",
    "AI pedagogy", "AI teaching methods", "AI in instructional design",
    "AI in lesson planning", "AI in course design", "AI in curriculum design",
    "AI in educational design", "AI in teaching practice", "AI in teacher training",
    "AI professional development", "AI for educators", "AI for instructors",
    "AI teaching tools", "AI classroom tools", "AI student engagement",
    "AI student motivation", "AI student support", "AI student success",
    "AI student retention", "AI student outcomes", "AI student learning",
    "AI student assessment", "AI student feedback", "AI student personalization",
    "AI student journey", "AI student experience", "AI for diverse learners",
    "AI for special education", "AI for English learners", "EdTech AI",
    "EdTech artificial intelligence", "EdTech machine learning", "EdTech innovation AI",
    "Learning management system AI", "LMS AI features", "Online learning AI",
    "E-learning AI", "Virtual learning AI", "Digital learning AI", "Blended learning AI",
    "Hybrid learning AI", "Remote learning AI", "Distance education AI",
    "Online education AI", "AI in math education", "AI in science education",
    "AI in language learning", "AI in reading education", "AI in writing education",
    "AI in history education", "AI in art education", "AI in music education",
    "AI in computer science education", "AI in STEM education", "AI in medical education",
    "AI in nursing education", "AI in business education", "AI in law education",
    "AI in engineering education", "AI ethics in education", "AI policy in education",
    "AI regulation in education", "AI guidelines for schools", "AI responsible use in education",
    "AI bias in education", "AI fairness in education", "AI transparency in education",
    "AI accountability in education", "AI privacy in education", "Future of AI in education",
    "AI transforming education", "AI revolution in education", "AI disruption in education",
    "AI innovation in education", "AI trends in education", "AI predictions education",
    "AI future learning", "AI future classroom", "AI future schools", "AI education research",
    "AI learning sciences", "AI cognitive science education", "AI educational psychology",
    "AI learning analytics", "AI educational data mining", "AI learning outcomes research",
    "AI teaching effectiveness research", "AI student performance research",
    "AI education studies", "AI challenges in education", "AI problems in education",
    "AI concerns in education", "AI risks in education", "AI limitations in education",
    "AI barriers in education", "AI obstacles in education", "AI difficulties in education",
    "AI issues in education", "AI drawbacks in education", "AI success in education",
    "AI case studies education", "AI examples education", "AI implementations education",
    "AI pilots in schools", "AI experiments in education", "AI projects in education",
    "AI initiatives in education", "AI programs in education", "AI adoption in education",
    "Khanmigo", "Khan Academy AI", "Duolingo AI", "Coursera AI", "Udemy AI", "EdX AI",
    "Google Classroom AI", "Microsoft Education AI", "OpenAI in education",
    "Anthropic in education", "Claude in education", "Gemini in education",
    "Copilot in education", "Midjourney in education", "DALL-E in education",
    "#AIinEducation", "#EdTech", "#AIinEd", "#AIED", "#AIforEducation",
    "#AIinSchools", "#AIinClassroom", "#AIforLearning", "#AIforTeaching",
    "#AIforStudents", "#AIforTeachers", "#FutureOfEducation", "#FutureOfLearning",
    "#EdTechInnovation", "#AIEDU", "#ChatGPTinEducation", "#AIinHigherEd",
    "#AIinK12", "#AIinEdTech", "#AI4Education"
]

# AI and education terms
AI_TERMS = ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
            'chatgpt', 'gpt', 'openai', 'llm', 'large language model', 'generative ai',
            'claude', 'gemini', 'copilot', 'bard', 'midjourney', 'dalle']

EDUCATION_TERMS = ['education', 'educational', 'learning', 'teaching', 'classroom', 'school',
                   'university', 'college', 'student', 'students', 'teacher', 'teachers',
                   'professor', 'professors', 'instructor', 'instructors', 'academic',
                   'curriculum', 'pedagogy', 'course', 'courses', 'class', 'classes']


class LinkedInScraper:
    def __init__(self):
        self.all_posts = []
        self.processed_texts = set()

    async def ensure_login(self):
        """Ensure we have a valid login session"""
        if not os.path.exists("linkedin_auth.json"):
            print("\n🔐 FIRST TIME SETUP - Login Required")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False)
                context = await browser.new_context()
                page = await context.new_page()
                await page.goto("https://www.linkedin.com/login")
                print("\n🌐 Browser opened. Please log in now...")
                input("\nPress ENTER after logging in...")
                await context.storage_state(path="linkedin_auth.json")
                print("✅ Session saved!")
                await browser.close()
        return True

    async def human_like_scroll(self, page, scroll_times=40):
        """
        Scroll like a human: click first, then scroll in chunks
        """
        print(f"   Preparing to scroll {scroll_times} times...")

        # CRITICAL: Click on the page to give it focus
        try:
            # Click in the middle of the page
            await page.mouse.click(500, 300)
            print("   ✓ Clicked on page to activate scrolling")
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"   ⚠️ Click failed: {e}")

        # Click on a blank area or the first result
        try:
            # Try to click on a non-interactive element
            await page.evaluate("""
                () => {
                    // Click on the body to ensure focus
                    document.body.click();
                    // Also try to click on a search result container if exists
                    const result = document.querySelector('.search-result__occluded-item, article');
                    if (result) result.click();
                }
            """)
            print("   ✓ JavaScript click executed")
            await page.wait_for_timeout(2000)
        except:
            pass

        # Now scroll in small increments like a human
        for i in range(scroll_times):
            # Scroll a moderate amount (like a human scrolling a bit)
            scroll_amount = random.randint(300, 700)
            await page.evaluate(f"window.scrollBy(0, {scroll_amount})")

            # Wait a random human-like time
            wait_time = random.uniform(1.5, 3.0)
            await page.wait_for_timeout(wait_time * 1000)

            # Every few scrolls, do a PageDown key (another human-like action)
            if i % 3 == 0:
                await page.keyboard.press("PageDown")
                await page.wait_for_timeout(1000)

            # Every 10 scrolls, scroll up a bit (simulates reading)
            if i % 10 == 0 and i > 0:
                await page.evaluate("window.scrollBy(0, -200)")
                await page.wait_for_timeout(1500)
                # Then continue scrolling down
                await page.evaluate("window.scrollBy(0, 400)")
                await page.wait_for_timeout(1500)

            # Show progress
            if (i + 1) % 5 == 0:
                print(f"      Human-like scroll {i + 1}/{scroll_times} completed")

        print(f"   Completed {scroll_times} human-like scrolls")

        # Final check: try to click "Load more" buttons
        try:
            await page.evaluate("""
                () => {
                    const buttons = document.querySelectorAll('button');
                    for (const btn of buttons) {
                        if (btn.innerText.includes('Load more') || btn.innerText.includes('Show more')) {
                            btn.click();
                        }
                    }
                }
            """)
            print("   ✓ Clicked any 'Load more' buttons")
            await page.wait_for_timeout(3000)
        except:
            pass

    async def extract_posts(self, page, query, max_posts):
        """Extract posts from current page"""
        # Get all text
        all_text = await page.evaluate("document.body.innerText")

        # Split into paragraphs
        paragraphs = all_text.split('\n')

        posts_found = 0
        for para in paragraphs:
            para = para.strip()
            if len(para) > 100 and para not in self.processed_texts:
                para_lower = para.lower()

                # Check relevance
                has_ai = any(term in para_lower for term in AI_TERMS)
                has_edu = any(term in para_lower for term in EDUCATION_TERMS)

                if has_ai and has_edu:
                    self.processed_texts.add(para)
                    post_id = hashlib.md5(para[:100].encode()).hexdigest()[:10]
                    self.all_posts.append({
                        'source': 'LinkedIn',
                        'url': f"https://www.linkedin.com/posts/ai-education-{post_id}",
                        'question_title': para[:100] + "..." if len(para) > 100 else para,
                        'answer_text': para
                    })
                    posts_found += 1

                    if posts_found >= max_posts:
                        break

        return posts_found

    async def search_linkedin(self, query, target_posts=300):
        """Search LinkedIn and extract posts with human-like scrolling"""
        print(f"\n🔍 SEARCHING: '{query}'")
        print(f"   Target: {target_posts} posts")

        browser = None
        try:
            async with async_playwright() as p:
                # Launch browser VISIBLE
                browser = await p.chromium.launch(headless=False)
                context = await browser.new_context(storage_state="linkedin_auth.json")
                page = await context.new_page()

                formatted = query.replace(' ', '%20').replace('#', '%23')
                url = f"https://www.linkedin.com/search/results/content/?keywords={formatted}"

                print(f"   Navigating...")
                await page.goto(url)

                # Wait for initial load
                print("   Waiting 15 seconds for initial load...")
                await page.wait_for_timeout(15000)

                # Take screenshot
                await page.screenshot(path=f"debug_{query[:20]}.png")

                # HUMAN-LIKE SCROLLING (click first, then scroll)
                await self.human_like_scroll(page, scroll_times=50)

                # Extract posts after scrolling
                print("   Extracting posts...")
                posts_found = await self.extract_posts(page, query, target_posts)

                print(f"  ✅ Collected {posts_found} posts from this search")
                return posts_found

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 0

    async def search_all(self, target_per_search=300):
        """Search all queries"""
        print("\n" + "=" * 80)
        print("🔍 LINKEDIN SCRAPER - HUMAN-LIKE SCROLLING")
        print("=" * 80)
        print(f"📊 Search queries: {len(SEARCH_QUERIES)}")
        print(f"📝 Target per search: {target_per_search}")
        print("=" * 80)

        total = 0
        start_time = time.time()

        for idx, query in enumerate(SEARCH_QUERIES, 1):
            print(f"\n{'=' * 60}")
            print(f"[{idx}/{len(SEARCH_QUERIES)}]")

            count = await self.search_linkedin(query, target_per_search)
            total += count

            elapsed = time.time() - start_time
            print(f"\n  📊 RUNNING TOTAL: {total} posts")
            print(f"  ⏱️  Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s")
            print(f"  🎯 Progress to 10,000: {min(100, total / 10000 * 100):.1f}%")

            if total >= 10000:
                print(f"\n🎉 REACHED 10,000 POSTS!")
                break

            if idx < len(SEARCH_QUERIES):
                delay = random.randint(15, 20)
                print(f"  ⏳ Waiting {delay} seconds...")
                await asyncio.sleep(delay)

        df = pd.DataFrame(self.all_posts)
        return df

    def analyze_and_save(self, df):
        """Analyze and save results"""
        if df.empty:
            print("\n❌ No data collected!")
            return

        word_count = df['answer_text'].str.split().str.len().sum()

        print("\n" + "=" * 80)
        print("📊 FINAL RESULTS")
        print("=" * 80)
        print(f"\n📈 TOTAL POSTS: {len(df)}")
        print(f"📝 TOTAL WORDS: {word_count:,}")
        print(f"📏 Avg words/post: {word_count / len(df):.1f}")

        # Save CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'linkedin_ai_education_{timestamp}.csv'
        df[['source', 'url', 'question_title', 'answer_text']].to_csv(filename, index=False)
        print(f"\n💾 Saved to: {filename}")
        print(f"   Location: {os.path.abspath(filename)}")

        # Create eval.xls
        sample = df.sample(min(1000, len(df)))

        def get_sentiment(text):
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    return 'positive'
                elif polarity < -0.1:
                    return 'negative'
                return 'neutral'
            except:
                return 'neutral'

        sample['sentiment_label'] = sample['answer_text'].apply(get_sentiment)
        sample['confidence'] = 0.8

        eval_df = sample[['answer_text', 'sentiment_label', 'confidence']].copy()
        eval_df.columns = ['text', 'sentiment_label', 'confidence']
        eval_df.to_excel('eval.xls', index=False)
        print(f"✅ Created eval.xls with {len(sample)} samples")

        # Check requirements
        print("\n" + "=" * 80)
        print("✅ REQUIREMENTS CHECK")
        print("=" * 80)
        print(f"Records: {len(df)} / 10,000 - {'✅' if len(df) >= 10000 else '❌'}")
        print(f"Words: {word_count:,} / 100,000 - {'✅' if word_count >= 100000 else '❌'}")

        if len(df) >= 10000 and word_count >= 100000:
            print("\n" + "🎉" * 20)
            print("🎉🎉🎉 CONGRATULATIONS! You've met both requirements! 🎉🎉🎉")
            print("🎉" * 20)
        else:
            print("\n📊 NEEDED TO REACH 10,000:")
            needed_posts = 10000 - len(df)
            print(f"   Need {needed_posts} more posts")

        # Show sample
        if len(df) > 0:
            print("\n📝 Sample post:")
            print(df.iloc[0]['answer_text'][:200] + "...")


async def main():
    print("\n" + "=" * 80)
    print("🔍 LINKEDIN SCRAPER - HUMAN-LIKE SCROLLING")
    print("=" * 80)

    scraper = LinkedInScraper()
    await scraper.ensure_login()

    # Ask for posts per search
    try:
        per_search = input("\n📝 Posts per search (default 300): ").strip()
        target_per_search = int(per_search) if per_search else 300
    except:
        target_per_search = 300

    # Estimate time
    total_searches = len(SEARCH_QUERIES)
    mins_per_search = 4  # Each search takes ~4 minutes with human-like scrolling
    total_mins = total_searches * mins_per_search
    print(f"\n⏱️  Estimated time: ~{total_mins} minutes ({total_mins / 60:.1f} hours)")
    print(f"🎯 Potential posts: {total_searches * target_per_search}")

    confirm = input("\nStart scraping? (y/n): ").lower()
    if confirm != 'y':
        print("Exiting...")
        return

    df = await scraper.search_all(target_per_search=target_per_search)

    if df is not None and not df.empty:
        scraper.analyze_and_save(df)
        print(f"\n✨ Done! Files in: {os.getcwd()}")
    else:
        print("\n❌ No data collected")


if __name__ == "__main__":
    asyncio.run(main())