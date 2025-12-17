import json
from openai import OpenAI
from config import OPENAI_API_KEY
import random

client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error: {e}")
        return None


# Load existing knowledge
with open('rag_knowledge.json', 'r') as f:
    knowledge = json.load(f)

# FILTER: Keep only top categories and limit items
ALLOWED_CATEGORIES = {
    'wisdom', 'inspiration', 'life', 'love', 'philosophy',
    'truth', 'happiness', 'hope', 'motivation', 'courage',
    'friendship', 'success', 'faith', 'time', 'dreams'
}

MAX_ITEMS_PER_CATEGORY = 500  # Limit to 500 best quotes per category

print(" Filtering dataset...")

# Filter and limit quotes
filtered_knowledge = {"morals": {}, "quotes": {}}

for category, items in knowledge.get('quotes', {}).items():
    # Only keep allowed categories
    if category not in ALLOWED_CATEGORIES:
        continue

    # Limit items per category
    if len(items) > MAX_ITEMS_PER_CATEGORY:
        items = random.sample(items, MAX_ITEMS_PER_CATEGORY)

    filtered_knowledge['quotes'][category] = items

# Copy morals as is (usually small)
filtered_knowledge['morals'] = knowledge.get('morals', {})

total_quotes = sum(len(items) for items in filtered_knowledge['quotes'].values())
print(f" Filtered to {total_quotes} quotes in {len(filtered_knowledge['quotes'])} categories")
print(f"   (Original: {sum(len(items) for items in knowledge.get('quotes', {}).values())} quotes)")

# Now add embeddings to filtered data
print("\n Adding embeddings to filtered data...")

processed = 0
total = total_quotes + sum(len(items) for items in filtered_knowledge['morals'].values())

# Process morals
for category, items in filtered_knowledge['morals'].items():
    new_items = []
    for item in items:
        if isinstance(item, dict):
            new_items.append(item)
        else:
            embedding = get_embedding(item)
            if embedding:
                new_items.append({'text': item, 'embedding': embedding})
                processed += 1
                if processed % 50 == 0:
                    print(f"   Progress: {processed}/{total}")
            else:
                new_items.append(item)

    filtered_knowledge['morals'][category] = new_items

# Process quotes
for category, items in filtered_knowledge['quotes'].items():
    new_items = []
    for item in items:
        if isinstance(item, dict):
            new_items.append(item)
        else:
            embedding = get_embedding(item)
            if embedding:
                new_items.append({'text': item, 'embedding': embedding})
                processed += 1
                if processed % 50 == 0:
                    print(f"   Progress: {processed}/{total}")
            else:
                new_items.append(item)

    filtered_knowledge['quotes'][category] = new_items

# Save
with open('rag_knowledge.json', 'w') as f:
    json.dump(filtered_knowledge, f, indent=2, ensure_ascii=False)

print(f"\n Migration completed!")
print(f"   Total embeddings created: {processed}")
print(f"   Estimated cost: ${processed * 0.00002:.2f}")