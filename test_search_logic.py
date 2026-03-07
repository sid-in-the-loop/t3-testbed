import asyncio
import os
import sys
import traceback
from pathlib import Path

# Ensure general_agent is in path
GA_DIR = Path(__file__).parent / "general_agent"
if str(GA_DIR) not in sys.path:
    sys.path.insert(0, str(GA_DIR))

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = GA_DIR / ".env"
    load_dotenv(env_path)
    print(f"Loaded .env from {env_path}")
    print(f"OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"SERPER_API_KEY present: {bool(os.getenv('SERPER_API_KEY'))}")
except ImportError:
    print("python-dotenv not installed")

# Turn on litellm debug before importing call_llm
try:
    import litellm
    # litellm._turn_on_debug() # Commenting out for cleaner output if key works
except ImportError:
    pass

from webwalkerqa.llm import call_llm
from webwalkerqa.search import web_search

async def test():
    try:
        print("\nTesting call_llm...")
        resp, p, o = await call_llm(
            messages=[{"role": "user", "content": "Say hello!"}],
            model="openai/gpt-4o-mini"
        )
        print(f"LLM Response: {resp}")
    except Exception:
        print("LLM Call Failed!")
        traceback.print_exc()
    
    try:
        print("\nTesting web_search...")
        search_res = await web_search("Polish actor who played Ray in Everybody Loves Raymond")
        print(f"Search Result (first 200 chars): {search_res[:200]}")
    except Exception:
        print("Web Search Failed!")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
