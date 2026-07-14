import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
JSON_OUTPUT_PATH = OUTPUT_DIR / "results.json"
CSV_OUTPUT_PATH = OUTPUT_DIR / "results.csv"
FAISS_INDEX_PATH = OUTPUT_DIR / "faiss_index.bin"

# LLM Config
GROQ_MODEL = "llama-3.3-70b-versatile"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
