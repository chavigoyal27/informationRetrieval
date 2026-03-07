"""
X Scraper - login to X and get cookies from browser developer tools
libraries to install: pip install twikit pandas textblob xlwt openpyxl httpx python-dotenv
"""

import asyncio
import pandas as pd
from datetime import datetime
import os
import random
import json
from textblob import TextBlob
from twikit import Client

# Your cookies
COOKIES_DICT = {
    "auth_token": "placeholder",
    "ct0": "placeholder",
    "twid": "placeholder",
    "guest_id": "placeholder"
}

# AI Terms (keeping full list)
AI_TERMS = [
    'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning', 'dl',
    'chatgpt', 'gpt', 'gpt4', 'gpt-4', 'openai', 'llm', 'large language model',
    'generative ai', 'genai', 'claude', 'anthropic', 'gemini', 'google ai',
    'copilot', 'microsoft copilot', 'bard', 'midjourney', 'dalle', 'dall-e',
    'stable diffusion', 'hugging face', 'transformers', 'neural networks',
    'nlp', 'natural language processing', 'computer vision', 'cv',
    'ai tools', 'ai assistant', 'ai tutor', 'intelligent tutoring',
    'adaptive learning', 'personalized learning', 'ai chatbot', 'chatbot',
    'llama', 'mistral', 'gemma', 'claude', 'perplexity', 'c.ai',
    'ai powered', 'ai-driven', 'ai-based', 'ai enhanced', 'ai augmented',
    'smart learning', 'intelligent system', 'cognitive computing',
    'neural network', 'deep neural network', 'transformer model',
    'gpt-3', 'gpt3', 'gpt-4o', 'claude 3', 'gemini advanced',
    'ai technology', 'artificial intelligence technology',
    'machine intelligence', 'computational intelligence',
    'automated learning', 'automated tutoring', 'automated teaching'
]

# Education Terms (keeping full list)
EDUCATION_TERMS = [
    'education', 'educational', 'learning', 'teaching', 'classroom', 'school',
    'university', 'college', 'student', 'students', 'teacher', 'teachers',
    'professor', 'professors', 'instructor', 'instructors', 'academic',
    'curriculum', 'pedagogy', 'course', 'courses', 'class', 'classes',
    'homework', 'assignment', 'exam', 'test', 'grade', 'grading',
    'study', 'studying', 'tutor', 'tutoring', 'lesson', 'lessons',
    'k12', 'k-12', 'highered', 'higher education', 'edtech', 'ed tech',
    'online learning', 'elearning', 'e-learning', 'distance learning',
    'remote learning', 'hybrid learning', 'blended learning',
    'professional development', 'lifelong learning', 'upskilling',
    'reskilling', 'vocational', 'training', 'workshop', 'seminar',
    'graduation', 'degree', 'diploma', 'certificate', 'accreditation',
    'faculty', 'staff', 'administration', 'dean', 'principal',
    'library', 'lab', 'laboratory', 'campus', 'dormitory',
    'primary school', 'secondary school', 'high school', 'middle school',
    'elementary school', 'preschool', 'kindergarten', 'graduate school',
    'medical school', 'law school', 'business school', 'engineering school',
    'classroom management', 'lesson plan', 'teaching method',
    'learning experience', 'student engagement', 'student success',
    'academic performance', 'learning outcomes', 'educational technology',
    'instructional design', 'learning management system', 'lms',
    'virtual classroom', 'virtual learning', 'online course', 'mooc',
    'flipped classroom', 'project-based learning', 'problem-based learning',
    'competency-based education', 'mastery learning', 'self-paced learning',
    'adult education', 'continuing education', 'executive education',
    'special education', 'inclusive education', 'differentiated instruction',
    'stem education', 'steam education', 'computer science education',
    'coding education', 'programming education', 'digital literacy',
    'information literacy', 'media literacy', 'data literacy',
    'study skills', 'critical thinking', 'problem solving',
    'collaborative learning', 'cooperative learning', 'peer learning',
    'group work', 'team project', 'research project', 'thesis', 'dissertation'
]


def generate_search_queries():
    """Generate search queries"""
    queries = []

    # Direct combinations
    for ai_term in AI_TERMS[:20]:
        for edu_term in EDUCATION_TERMS[:20]:
            queries.append(f"{ai_term} {edu_term}")

    # Phrases
    phrases = [
        "AI in education", "AI for learning", "AI teaching", "AI classroom",
        "ChatGPT in education", "machine learning education", "AI students",
        "AI teachers", "AI in schools", "AI in universities", "AI tutoring",
        "AI homework help", "AI assessment", "AI grading", "AI curriculum",
        "AI personalized learning", "adaptive learning AI", "AI study assistant",
        "AI in higher education", "AI in K-12", "future of AI in education",
        "AI education technology", "EdTech AI", "AI learning tools"
    ]
    queries.extend(phrases)

    # Remove duplicates
    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)

    return unique_queries


SEARCH_QUERIES = generate_search_queries()
print(f"📊 Generated {len(SEARCH_QUERIES)} search queries")


class XRateLimitScraper:
    def __init__(self):
        self.client = None
        self.tweets = []
        self.processed_ids = set()
        self.rate_limit_hits = 0
        self.consecutive_errors = 0

    async def init(self):
        """Initialize with cookies"""
        print("\n📂 Initializing...")
        self.client = Client('en-US')
        try:
            self.client.set_cookies(COOKIES_DICT)
            print("✅ Cookies set")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False

    async def search_tweets(self, query, max_tweets=50):
        """Search with rate limit handling"""
        try:
            tweets = await self.client.search_tweet(query, 'Latest', count=40)
            collected = 0

            for tweet in tweets:
                if tweet.id in self.processed_ids:
                    continue

                self.processed_ids.add(tweet.id)
                text = tweet.full_text

                if len(text.split()) < 3:
                    continue

                title = text[:100] + "..." if len(text) > 100 else text

                self.tweets.append({
                    'source': 'X (Twitter)',
                    'url': f"https://twitter.com/i/web/status/{tweet.id}",
                    'question_title': title,
                    'answer_text': text
                })

                collected += 1
                if collected >= max_tweets:
                    break

            # Reset error counter on success
            self.consecutive_errors = 0
            return collected

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                self.rate_limit_hits += 1
                self.consecutive_errors += 1
                print(f"   ⚠️ RATE LIMIT HIT #{self.rate_limit_hits}")

                # Exponential backoff - wait longer each time
                wait_time = min(300, 30 * (2 ** self.consecutive_errors))  # Max 5 minutes
                print(f"   ⏳ Waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)

                # Try to refresh cookies or login again
                if self.consecutive_errors >= 3:
                    print("   🔄 Too many errors, reinitializing...")
                    await self.init()
                    self.consecutive_errors = 0
            else:
                print(f"   ❌ Error: {error_msg[:100]}")

            return 0

    async def scrape(self, target_tweets=10000):
        """Main scrape with rate limit awareness"""
        if not await self.init():
            return None

        total = 0
        start_time = datetime.now()
        queries_used = 0

        # Slower, more deliberate pace
        for i, query in enumerate(SEARCH_QUERIES, 1):
            if total >= target_tweets:
                break

            queries_used += 1
            remaining = target_tweets - total
            max_per_query = min(30, remaining)  # Reduced to 30 per query

            print(f"\n[{i}/{len(SEARCH_QUERIES)}] Total: {total}/{target_tweets}")
            print(f"🔍 {query[:50]}...")

            collected = await self.search_tweets(query, max_per_query)

            if collected > 0:
                total += collected

                # Progress update
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                rate = total / elapsed if elapsed > 0 else 0
                print(f"📊 Got {collected} tweets | Total: {total} | Rate: {rate:.1f}/min")
                print(f"🎯 {min(100, total / target_tweets * 100):.1f}% complete")

                # Variable delay based on success
                if self.rate_limit_hits > 0:
                    # If we've hit rate limits, go slower
                    delay = random.uniform(10, 20)
                else:
                    delay = random.uniform(5, 10)

                print(f"⏳ Next in {delay:.0f}s...")
                await asyncio.sleep(delay)
            else:
                # If no tweets collected, shorter delay
                await asyncio.sleep(2)

        print(f"\n📊 Used {queries_used} queries, hit rate limit {self.rate_limit_hits} times")
        return pd.DataFrame(self.tweets)

    def save_results(self, df):
        """Save results"""
        if df.empty:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        total_tweets = len(df)
        total_words = df['answer_text'].str.split().str.len().sum()

        print("\n" + "=" * 80)
        print("📊 FINAL STATISTICS")
        print("=" * 80)
        print(f"📈 TOTAL TWEETS: {total_tweets}")
        print(f"📝 TOTAL WORDS: {total_words:,}")

        # Save
        filename = f'x_ai_education_{timestamp}.csv'
        df[['source', 'url', 'question_title', 'answer_text']].to_csv(filename, index=False)
        print(f"\n💾 Saved to: {filename}")

        # Create eval.xls
        if len(df) >= 100:
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

            try:
                eval_df.to_excel('eval.xls', index=False, engine='xlwt')
                print("✅ Created eval.xls")
            except:
                eval_df.to_csv('eval.csv', index=False)
                print("✅ Created eval.csv")

        # Check requirements
        print("\n" + "=" * 80)
        print("✅ REQUIREMENTS CHECK")
        print("=" * 80)
        print(f"Records: {total_tweets} / 10,000 - {'✅' if total_tweets >= 10000 else '❌'}")
        print(f"Words: {total_words:,} / 100,000 - {'✅' if total_words >= 100000 else '❌'}")


async def main():
    print("\n" + "=" * 80)
    print("🐦 X SCRAPER - WITH RATE LIMIT HANDLING")
    print("=" * 80)

    scraper = XRateLimitScraper()

    target = 10000
    print(f"\n🎯 Target: {target} tweets")
    print("⚠️  Going slow to avoid rate limits...")

    df = await scraper.scrape(target_tweets=target)

    if df is not None and not df.empty:
        scraper.save_results(df)
        print(f"\n✨ Done!")
    else:
        print("\n❌ Failed")


if __name__ == "__main__":
    asyncio.run(main())