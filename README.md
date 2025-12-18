# Story Moral & Quote Extraction Agent

An AI-powered system that extract meaningful moral lessons and memorable
quotes from stories using RAG (Retrieval-Augmented Generation) with 
semantics embeddings

---

## What Does It Do?

- **Extract Morals**: Identifies life lessons and wisdom from stories
- **Extract Quotes**: Finds memorable and quotable lines
- **Smart Categorization**: Uses AI to categorize morals and quotes
- **RAG-Powered**: Uses embeddings for semantic similarity matching
- **Learning System**: Builds knowledge base over time from processed stories

---

## Requirements

- Python 3.14+
- OpenAI API key

---

## Quick Start  

### 1. Clone & Setup

```bash
# Clone the repository
git clone
cd agent


# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
# OR
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `config.py` file:

```python
OPENAI_API_KEY = "your-api-key-here"
```

### 3. Run the Agent

```bash
# Basic usage(uses default story file)
python main.py

# With custom story file
python main.py path/to/story.txt path/to/output.json
```

# Generate embeddings(if updating dataset)

```bash
python migrate_embeddings.py

```

## Running Tests

```bash
pytest test_agent.py -v 
```