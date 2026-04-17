"""Retrieval service: Wikipedia + NewsAPI + SerpAPI"""

from __future__ import annotations

import os
import requests


# ---------------------------
# CONFIG (ENV VARIABLES)
# ---------------------------
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")


# ---------------------------
# WIKIPEDIA
# ---------------------------
def fetch_wikipedia(query: str, top_k: int = 2):
    results = []

    try:
        search_url = "https://en.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }

        response = requests.get(search_url, params=params, timeout=5)
        data = response.json()

        for i, item in enumerate(data.get("query", {}).get("search", [])[:top_k]):
            title = item["title"]

            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            summary_res = requests.get(summary_url, timeout=5)

            if summary_res.status_code == 200:
                summary_data = summary_res.json()

                results.append({
                    "text": summary_data.get("extract", ""),
                    "source": f"https://en.wikipedia.org/wiki/{title}",
                    "score": round(1 / (i + 1), 2),
                })

    except Exception:
        pass

    return results


# ---------------------------
# NEWS API
# ---------------------------
def fetch_news(query: str, top_k: int = 2):
    results = []

    if not NEWS_API_KEY:
        return results

    try:
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "pageSize": top_k,
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        for i, article in enumerate(data.get("articles", [])[:top_k]):
            results.append({
                "text": article.get("title", ""),
                "source": article.get("url", ""),
                "score": round(1 / (i + 1), 2),
            })

    except Exception:
        pass

    return results


# ---------------------------
# SERP API (GOOGLE SEARCH)
# ---------------------------
def fetch_serp(query: str, top_k: int = 2):
    results = []

    if not SERP_API_KEY:
        return results

    try:
        url = "https://serpapi.com/search.json"

        params = {
            "q": query,
            "api_key": SERP_API_KEY,
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        for i, item in enumerate(data.get("organic_results", [])[:top_k]):
            results.append({
                "text": item.get("title", ""),
                "source": item.get("link", ""),
                "score": round(1 / (i + 1), 2),
            })

    except Exception:
        pass

    return results


# ---------------------------
# MAIN RETRIEVAL FUNCTION
# ---------------------------
def retrieve_evidence(query: str, top_k: int = 5):
    """Combine multiple sources into unified evidence list"""

    if not query.strip():
        return {"evidence": []}

    evidence = []

    # Fetch from all sources
    wiki = fetch_wikipedia(query)
    news = fetch_news(query)
    serp = fetch_serp(query)

    # Combine results
    evidence.extend(wiki)
    evidence.extend(news)
    evidence.extend(serp)

    # If nothing found → fallback
    if not evidence:
        return {
            "evidence": [
                {
                    "text": "No external sources found for this query.",
                    "source": "system",
                    "score": 0.0,
                }
            ]
        }

    # Sort by score (highest first)
    evidence = sorted(evidence, key=lambda x: x["score"], reverse=True)

    # Take top_k best results
    return {"evidence": evidence[:top_k]}
