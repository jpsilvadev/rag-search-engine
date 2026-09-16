import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
model = "openrouter/free"


def spell_correct(query: str) -> str:
    # NOTE: "openrouter/free" auto-routes to whichever free model is available,
    # which can occasionally return unrelated/garbage output (e.g. a safety
    # classifier label instead of a corrected query). No validation is done
    # on the result here, so bad output currently passes through as-is.
    sys_prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        return query
    return content.strip().strip('"')


def rewrite_query(query: str) -> str:
    sys_prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        return query
    return content.strip().strip('"')


def enhance_query(query: str, method: str | None = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case _:
            return query
