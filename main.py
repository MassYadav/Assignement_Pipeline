import asyncio
import logging
import json
import csv
from config import DATA_DIR, JSON_OUTPUT_PATH, CSV_OUTPUT_PATH
from data_loader import load_contracts
from llm_extractor import extract_contract_details
from semantic_search import SemanticSearch

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def process_contracts():
    logger.info("Starting contract processing pipeline...")

    # 1. Load PDFs
    logger.info(f"Loading PDFs from {DATA_DIR}")
    contracts = load_contracts(DATA_DIR)
    
    if not contracts:
        logger.error("No contracts found to process. Please place PDF files in the data/ directory.")
        return

    # 2. Process sequentially via Groq
    results_list = []
    semantic_db = SemanticSearch()

    for contract_id, text in contracts.items():
        logger.info(f"Processing contract: {contract_id}")
        extraction = await extract_contract_details(text)
        
        if extraction:
            extraction_dict = extraction.model_dump()
            
            # Save for results
            result_entry = {
                "contract_id": contract_id,
                **extraction_dict
            }
            results_list.append(result_entry)
            
            # 3. Index clauses into semantic search
            semantic_db.index_clauses(contract_id, extraction_dict)
        else:
            logger.warning(f"Failed to extract details for {contract_id}")

    # 4. Save results to JSON and CSV
    if results_list:
        # Save JSON
        with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=4)
        logger.info(f"Saved results to {JSON_OUTPUT_PATH}")
        
        # Save CSV
        csv_columns = ["contract_id", "summary", "termination_clause", "confidentiality_clause", "liability_clause"]
        with open(CSV_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for data in results_list:
                writer.writerow(data)
        logger.info(f"Saved results to {CSV_OUTPUT_PATH}")
    
    # 5. Run a sample search query
    sample_query = "What happens if the contract is breached?"
    logger.info(f"Running sample search query: '{sample_query}'")
    search_results = semantic_db.search(sample_query, top_k=2)
    
    for i, res in enumerate(search_results):
        logger.info(f"Search Result {i+1}:")
        logger.info(f"  Contract: {res['contract_id']}")
        logger.info(f"  Clause Type: {res['clause_type']}")
        logger.info(f"  Text Snippet: {res['text'][:200]}...")

if __name__ == "__main__":
    asyncio.run(process_contracts())
