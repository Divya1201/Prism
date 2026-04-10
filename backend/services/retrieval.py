from __future__ import annotations

import requests


def retrieve_evidence(query: str, top_k: int = 3):
    """Retrieve real-world evidence using Wikipedia API"""

    try:
        # Search Wikipedia
        search_url = "https://en.wikipedia.org/w/api.php"

        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }

        search_res = requests.get(search_url, params=search_params, timeout=5)
        search_data = search_res.json()

        results = search_data.get("query", {}).get("search", [])[:top_k]

        evidence = []

        for item in results:
            title = item["title"]

            # Fetch summary
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            summary_res = requests.get(summary_url, timeout=5)

            if summary_res.status_code == 200:
                summary_data = summary_res.json()
                text = summary_data.get("extract", "")

                evidence.append({
                    "text": text,
                    "source": f"https://en.wikipedia.org/wiki/{title}",
                    "score": 0.9  # static for now
                })

        return {"evidence": evidence}

    except Exception:
        return {
            "evidence": [
                {
                    "text": "Unable to retrieve external sources at the moment.",
                    "source": "system",
                    "score": 0.0
                }
            ]
        }
