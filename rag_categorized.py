import json
import os
from collections import defaultdict


class SimpleRAG:
    """Simple RAG for better categorization"""

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

    def _get_keywords(self, text):
        """Extract keywords from text"""
        stop_words = {'the', 'is', 'are', 'was', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'that', 'it'}
        words = text.lower().replace('.', '').replace(',', '').split()
        return [w for w in words if w not in stop_words and len(w) > 3]

    def find_best_category(self, text, original_category, item_type="moral"):
        """Find best matching category from past data"""

        db_key = "morals" if item_type == "moral" else "quotes"
        past_data = self.knowledge.get(db_key, {})

        if not past_data:
            return original_category, 0, "No past data"

        keywords = self._get_keywords(text)
        category_scores = defaultdict(int)

        # Score each category based on keyword matches
        for category, items in past_data.items():
            for item in items:
                item_keywords = self._get_keywords(item)
                matches = set(keywords) & set(item_keywords)
                category_scores[category] += len(matches)

        if not category_scores:
            return original_category, 0, "No matches found"

        # Get best category
        best_category = max(category_scores, key=category_scores.get)
        confidence = min(90, category_scores[best_category] * 20)

        # Only use if confidence is high
        if confidence >= 40:
            return best_category, confidence, f"Matched with past {item_type}s"

        return original_category, 0, "Low confidence"

    def add_to_knowledge(self, text, category, item_type="moral"):
        """Add item to knowledge base"""
        db_key = "morals" if item_type == "moral" else "quotes"

        if db_key not in self.knowledge:
            self.knowledge[db_key] = {}

        if category not in self.knowledge[db_key]:
            self.knowledge[db_key][category] = []

        # Avoid duplicates
        if text not in self.knowledge[db_key][category]:
            self.knowledge[db_key][category].append(text)
            self._save_db()

    def improve_categories(self, items, item_type="moral"):
        """Apply RAG to improve categories"""
        improved = []

        for item in items:
            text = item['moral'] if item_type == "moral" else item['quote']
            original_cat = item['category']

            # Find better category
            better_cat, conf, reason = self.find_best_category(text, original_cat, item_type)

            # Update if improved
            if better_cat != original_cat and conf > 0:
                item['category'] = better_cat
                item['rag_improved'] = True
                item['rag_confidence'] = conf
                item['original_category'] = original_cat
            else:
                item['rag_improved'] = False

            # Save to knowledge base
            self.add_to_knowledge(text, item['category'], item_type)

            improved.append(item)

        return improved