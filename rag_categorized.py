import json
import os

from openai import OpenAI

from config import OPENAI_API_KEY


class SimpleRAG:
    """Embedding-based RAG with semantic similarity"""

    def __init__(self, db_path="rag_knowledge.json"):
        self.db_path = db_path
        self.knowledge = self._load_db()
        self.client = OpenAI(api_key = OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"

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

    def _get_embedding(self, text):
        """Get embedding vector for text"""
        try:
            response = self.client.embeddings.create(
                model = self.embedding_model,
                input = text[:8000]    # truncate if too long
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0

        dot_product = sum(a * b for a,b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)

    # Retrieval for pre-context
    def get_context_examples(self, story_text, item_type="moral", top_n = 10):
        """
        Retrieve semantically similar examples based on story

        Args:
            story_text: The input story text
            item_type: "moral" or "quote"
            top_n: Number of most relevant examples to retrieve
        """

        db_key = "morals" if item_type == "moral" else "quotes"
        past_data = self.knowledge.get(db_key, {})

        if not past_data:
            return ""

        # Get story embedding
        story_embedding = self._get_embedding(story_text)
        if not story_embedding:
            return ""

        # Calculate similarity for all items
        similarities = []

        for category, items in past_data.items():
            for item in items:
                # Check if item has embedding
                if isinstance(item, dict) and 'embedding' in item:
                    text = item['text']
                    item_embedding = item['embedding']
                else:
                    # Old format - just text , skip for now
                    continue

                # Calculate similarity
                sim = self._cosine_similarity(story_embedding, item_embedding)
                similarities.append({
                    'text': text,
                    'category': category,
                    'similarity': sim
                })

        # sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Take top N
        top_examples = similarities[:top_n]

        if not top_examples:
            return ""

        # Build context
        context = f"\nMost relevant {item_type} examples from similar stories:\n\n"

        for ex in top_examples:
            text = ex['text'][:60] + "..." if len(ex['text']) > 60 else ex['text']
            context += f"Example: '{text}' -> category {ex['category']}\n"

        context += "\nUse these relevant examples as guidance.\n"
        return context


    def add_to_knowledge(self, text, category, item_type="moral"):
        """Add item to knowledge base with embedding"""
        db_key = "morals" if item_type == "moral" else "quotes"

        if db_key not in self.knowledge:
            self.knowledge[db_key] = {}

        # Handle list categories
        if isinstance(category, list):
            category_key = ",".join(category)

        else:
            category_key = category

        if category_key not in self.knowledge[db_key]:
            self.knowledge[db_key][category_key] = []

        # Check if already exists
        existing_texts = [
            item['text'] if isinstance(item, dict) else item
            for item in self.knowledge[db_key][category_key]
        ]

        if text in existing_texts:
            return    # Already exists

        # Get embedding
        embedding = self._get_embedding(text)

        if embedding:
            # Save as dict with embedding
            self.knowledge[db_key][category_key].append({
                'text' : text,
                'embedding' : embedding
            })
            self._save_db()

