import json
import os
from collections import defaultdict


class SimpleRAG:
    """Standard RAG with pre-context post-processing"""

    def __init__(self, db_path="rag_knowledge.json"):
        self.db_path = db_path
        self.knowledge = self._load_db()

    def _load_db(self):
        """Load knowledge base"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"morals": {}, "quotes": {}}

    def _save_db(self):
        """Save knowledge base"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge, f, indent=2, ensure_ascii=False)

    # def _get_keywords(self, text):
    #     """Extract keywords from text"""
    #     if not isinstance(text, str) or not text.strip():
    #         return []
    #     stop_words = {'the', 'is', 'are', 'was', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'that', 'it'}
    #     words = text.lower().replace('.', '').replace(',', '').split()
    #     return [w for w in words if w not in stop_words and len(w) > 3]


    # Retrieval for pre-context
    def get_context_examples(self, item_type="moral", max_per_category = 2):
        """
        Retrieve past examples to provide as context to AI
        This is the RETRIEVAL step of standard RAG
        """
        import random

        db_key = "morals" if item_type == "moral" else "quotes"
        past_data = self.knowledge.get(db_key, {})

        if not past_data:
            return ""

        # Build context string with examples from each category
        context = f"\nHere are some past {item_type} categorization examples to guide you:\n\n"

        # Limit categories to prevent huge context
        categories = list(past_data.keys())

        if len(categories) > 10:
            categories = random.sample(categories, 10)

        for category in categories:
            items = past_data[category]
            if items:
                # Take random examples from each category
                examples = random.sample(items, min(max_per_category, len(items)))   # Random 3

                for example in examples:
                    # TRUNCATE long quotes
                    text = example[:60] + "..." if len(example) > 60 else example
                    context += f"Example: '{text}' → Category: {category}\n"

        context += "\nUse these examples as guidance for consistent categorization.\n"
        return context


    def add_to_knowledge(self, text, category, item_type="moral"):
        """Add item to knowledge base"""
        db_key = "morals" if item_type == "moral" else "quotes"

        if db_key not in self.knowledge:
            self.knowledge[db_key] = {}

        if isinstance(category, list):
            category_key = ",".join(category)

        else:
            category_key = category

        if category_key not in self.knowledge[db_key]:
            self.knowledge[db_key][category_key] = []

        # Avoid duplicates
        if text not in self.knowledge[db_key][category_key]:
            self.knowledge[db_key][category_key].append(text)
            self._save_db()

