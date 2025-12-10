from rag_categorized import SimpleRAG
from openai import OpenAI
from pydantic import BaseModel
from schema import MoralSchema, ResponseSchema
import json
import os
import sys
from config import logger

from agent import Agent
from config import OPENAI_API_KEY
from pydantic import BaseModel

# Helper Functions
def get_story(path: str):
    with open(path, "r") as file:
        return file.read()

def write_to_file(data: list[BaseModel], output_path: str):
    morals = data["morals"]
    quotes = data["quotes"]
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"morals": morals, "quotes": quotes}, file, indent=4, ensure_ascii=False)

        
def main():

    # Command line arguments
    story_path = sys.argv[1] if len(sys.argv) > 1 else "file/story.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "file/output.json"

    logger.info(f"Reading: {story_path}")
    logger.info(f"Output: {output_path}")

    # initialize RAG
    rag = SimpleRAG()
    logger.info(f"RAG Knowledge Base loaded")

    agent = Agent(open_api_key=OPENAI_API_KEY)
    story = get_story(path = story_path)
    result = agent.run(story)

    logger.info("Applying RAG categorization...")
    moral_dicts = [m.model_dump() for m in result['morals']]
    quote_dicts = [q.model_dump() for q in result['quotes']]


    improved_morals = rag.improve_categories(moral_dicts, "moral")
    improved_quotes = rag.improve_categories(quote_dicts, "quote")

    result['morals'] = improved_morals
    result['quotes'] = improved_quotes

    rag_improved_morals = sum(1 for m in improved_morals if m.get('rag_improved'))
    rag_improved_quotes = sum(1 for q in improved_quotes if q.get('rag_improved'))
    logger.info(f"RAG improved {rag_improved_morals} morals and {rag_improved_quotes} quotes")

    write_to_file(result, output_path)
    logger.info(f"Output saved successfully")
    return result

if __name__ == "__main__":
    main()