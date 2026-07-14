import logging
import json
from pydantic import BaseModel, Field
import instructor
from groq import AsyncGroq
import asyncio
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

class ContractExtraction(BaseModel):
    summary: str = Field(..., description="Concise 100-150 word summary highlighting purpose, key obligations, and notable risks/penalties.")
    termination_clause: str = Field(..., description="The termination clause extracted from the contract. If none, output 'Not found'.")
    confidentiality_clause: str = Field(..., description="The confidentiality clause extracted from the contract. If none, output 'Not found'.")
    liability_clause: str = Field(..., description="The liability clause extracted from the contract. If none, output 'Not found'.")

# Initialize the async Groq client with instructor
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set. LLM extraction will fail.")
    client = None
else:
    client = instructor.from_groq(AsyncGroq(api_key=GROQ_API_KEY))

async def extract_contract_details(text: str) -> ContractExtraction | None:
    """Uses Groq and instructor to extract structured fields from the contract text."""
    if not client:
        logger.error("Groq client not initialized (missing API key).")
        return None

    # Chunking: llama3-70b-8192 has a context limit of 8192 tokens. 
    # For safety, we truncate the text to approximately 20,000 characters.
    max_chars = 20000 
    if len(text) > max_chars:
        logger.info(f"Text too long ({len(text)} chars), truncating to {max_chars} chars.")
        text = text[:max_chars]

    try:
        extraction = await client.chat.completions.create(
            model=GROQ_MODEL,
            response_model=ContractExtraction,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal AI assistant. Extract the requested clauses and provide a concise summary from the given contract text."
                },
                {
                    "role": "user",
                    "content": f"Please extract the details from the following contract:\n\n{text}"
                }
            ],
            temperature=0.0
        )
        return extraction
    except Exception as e:
        logger.error(f"Error extracting details with Groq: {e}")
        return None
