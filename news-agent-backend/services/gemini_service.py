import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def summarize_article(title: str, snippet: str):
    """
    Returns 2-3 sentence summary for one article.
    """
    if not OPENAI_API_KEY:
        return snippet[:220] if snippet else ""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a news summarizer for a corporate dashboard."
                },
                {
                    "role": "user",
                    "content": f"""
Write EXACTLY 2 to 3 short sentences in very simple English.

Title: {title}
Text: {snippet}

Rules:
- 2 or 3 sentences only
- No bullet points
- No emojis
- No opinions
- Mention what happened and why it matters
"""
                }
            ],
            temperature=0.3,
            max_tokens=150,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("SHORT SUMMARY ERROR:", e)
        return snippet[:220] if snippet else ""


def summarize_executive(articles):
    """
    Executive summary (6-8 bullet points).
    """
    if not OPENAI_API_KEY:
        return "Summary unavailable."

    try:
        titles = "\n".join([f"- {a.get('title','')}" for a in articles[:12]])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are writing an executive news digest for senior leadership."
                },
                {
                    "role": "user",
                    "content": f"""
Input headlines:
{titles}

Write 6 to 8 bullet points summarizing the major themes.

Rules:
- Bullet points only
- One line per bullet
- Simple business language
- No speculation
"""
                }
            ],
            temperature=0.4,
            max_tokens=300,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("EXECUTIVE SUMMARY ERROR:", e)
        return "Executive summary failed."

def summarize_article_deep(title: str, snippet: str):
    """
    Structured deep summary using simple language and clear sections.
    Optimized for dashboard readability.
    """

    if not OPENAI_API_KEY:
        return snippet[:550] if snippet else "API key missing."

    if not snippet or len(snippet.strip()) < 40:
        return "Not enough article text available."

    try:
        # Prevent excessive token usage
        if len(snippet) > 5000:
            snippet = snippet[:5000]

        prompt = f"""
Article:
Title: {title}
Text: {snippet}

Create a structured news briefing using these EXACT headings:

TL;DR:
Give 2-3 very short sentences explaining the main news covering everhting main that has happened so busy users can scan everything in one go 

What Happened:
Explain the event in simple words.

Why It Matters:
Explain why this is important for people, business, or society.

What’s Next:
Explain expected future steps if mentioned.

STRICT RULES:
- Use VERY simple English yet keep it formal
- Each sentence must be SHORT but try keeping it around 2-3 lines
- Maximum 3 sentences per section
- Avoid difficult or corporate words but keep it formal
- Avoid repetition
- Do NOT invent facts
- Make it understandable in one quick read



"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You explain news clearly to busy professionals who need fast understanding."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=250,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("DEEP SUMMARY ERROR:", e)
        return "Deep summary generation failed."