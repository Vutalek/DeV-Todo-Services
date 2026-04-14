import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("ROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

DEFAULT_CANDIDATE_MODELS = [
    "google/gemini-2.5-flash-lite",  # 0.10, 0.40
    "deepseek/deepseek-v3.2",  # 0.26, 0.38
    "openai/gpt-4o-mini",  # 0.15, 0.60
]

DEFAULT_JUDGE_MODELS = [
    "openai/gpt-4o-mini",  # 0.15, 0.60
    "google/gemini-3-flash-preview",  # 0.50, 3
]

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2500
DEFAULT_JUDGE_MAX_TOKENS = 2500

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
DEFAULT_CANDIDATE_PROMPT_PATH = PROMPTS_DIR / "candidate_system_prompt.txt"
DEFAULT_JUDGE_PROMPT_PATH = PROMPTS_DIR / "judge_system_prompt.txt"
