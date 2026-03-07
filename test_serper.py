import requests
import json
import os

def test_search():
    api_key = "10fceeb603e65046b745dcc6baa9df6ed7cebce7"
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": "who wrote pride and prejudice", "num": 10}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        print("Success!")
        print(json.dumps(data, indent=2)[:500])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_search()
