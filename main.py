from rag_categorized import SimpleRAG
from pydantic import BaseModel
import json
import os
import sys
from config import logger

from agent import Agent
from config import OPENAI_API_KEY

# Helper Functions
def get_story(path: str):
    with open(path, "r") as file:
        return file.read()

def write_to_file(data: list[BaseModel], output_path: str):

    morals = [m.model_dump() if hasattr(m, 'model_dump') else m for m in data["morals"]]
    quotes = [q.model_dump() if hasattr(q, 'model_dump') else q for q in data["quotes"]]
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"morals": morals, "quotes": quotes}, file, indent=4, ensure_ascii=False)

        
def main():

    # Command line arguments
    story_path = sys.argv[1] if len(sys.argv) > 1 else "file/story1.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "file/output1.json"

    logger.info(f"Reading: {story_path}")
    logger.info(f"Output: {output_path}")

    # initialize RAG
    rag = SimpleRAG()
    logger.info(f"RAG Knowledge Base loaded")

    if rag.knowledge.get("morals"):
        moral_count = sum(len(items) for items in rag.knowledge["morals"].values())
        logger.info(f"Found {moral_count} past morals in knowledge base")

    if rag.knowledge.get("quotes"):
        quote_count = sum(len(items) for items in rag.knowledge["quotes"].values())
        logger.info(f"Found {quote_count} past quotes in knowledge base")


    agent = Agent(open_api_key=OPENAI_API_KEY, rag=rag)

    story = get_story(path = story_path)
    logger.info(f"Story length: {len(story)} characters")

    logger.info("Generating morals and quotes with RAG context...")


    result = agent.run(story)

    logger.info(f"Extracted {len(result['morals'])} morals and {len(result['quotes'])} quotes")

    # ======== Save to knowledge base (no post-processing!)
    logger.info("Saving to knowledge base...")
    for moral in result["morals"]:
        m = moral.model_dump() if hasattr(moral, 'model_dump') else moral
        rag.add_to_knowledge(m['moral'], m['category'], "moral")

    for quote in result["quotes"]:
        q = quote.model_dump() if hasattr(quote, 'model_dump') else quote
        rag.add_to_knowledge(q['quote'], q['category'], "quote")

    logger.info(f"Total morals: {len(result['morals'])}, Total quotes: {len(result['quotes'])}")

    # ====== Write Output ======
    write_to_file(result, output_path)
    logger.info(f"Output saved successfully to {output_path}")

    return result

if __name__ == "__main__":
    main()