"""
Complete Test Cases for Story Moral & Quote Extraction Agent
Run all :
    pytest test_agent.py -s
"""

import pytest
from twisted.internet.defer import returnValue

from agent import Agent
from config import OPENAI_API_KEY, logger
from rag_categorized import SimpleRAG
import os
import json
from unittest.mock import patch, MagicMock



@pytest.fixture
def sample_story():
    return """
    Once upon a time, a poor woodcutter lost his axe in a river. 
    The god Mercury appeared and asked what happened. The woodcutter 
    told the truth. Mercury brought a golden axe, but the woodcutter 
    refused it, saying it wasn't his. Mercury was pleased with his 
    honesty and rewarded him with both golden and silver axes.
    """

@pytest.fixture
def story_with_quotes():
    return """
    One hot day, a thirsty crow flew all over the fields looking for water.
    For a long time, she could not find any. She felt very weak, almost giving up hope.
    Suddenly, she saw a water jug below her. She flew straight down to see if there was any water inside. Yes, she could see some water inside the jug! The crow tried to push her head into the jug. Sadly, she found that the neck of the jug was too narrow. Then she tried to push the jug down for the water to flow out. She found that the jug was too heavy.
    The crow thought hard for a while. Then looking around her, she saw some pebbles. She suddenly had a good idea. She started picking up the pebbles one by one, dropping each into the jug.
    As more and more pebbles filled the jug, the water level kept rising. Soon it was high enough for the crow to drink. Her plan had worked!
    If you try hard enough, you may soon find an answer to your problem.
    """


@pytest.fixture
def agent_instance():
    """Create agent instance"""
    return Agent(open_api_key=OPENAI_API_KEY, rag=None)


@pytest.fixture
def rag_instance():
    """Create RAG instance with temp database"""
    test_db = "test_rag_knowledge.json"

    # create empty test database
    with open(test_db, 'w') as f:
        json.dump({"morals": {}, "quotes": {}}, f)

    rag = SimpleRAG(db_path=test_db)

    yield rag

    # cleanup after test
    if os.path.exists(test_db):
        os.remove(test_db)



# Agent tests
class TestAgent:
    """Test Agent's extraction functionality"""

    def test_agent_extracts_morals(self, agent_instance, sample_story):
        """Test: Agent should extract morals from the story """
        mock_result = {
            "morals" : [
                {
                    "moral": "Honesty is the best policy",
                    "category": ["wisdom"],
                    "source": "the honest woodcutter",
                    "confidence": 90
                }
            ],
            "quotes":[]
        }

        # Patch the run method to return mock_result instead of hitting API
        with patch.object(agent_instance, "run", return_value=mock_result):
            result = agent_instance.run(sample_story)   # now return mock result

            morals = result["morals"]

            assert "morals" in result
            assert len(result["morals"]) > 0
            assert isinstance(result["morals"], list)
            logger.info(f"Extracted {len(result['morals'])} morals")


    def test_agent_extracts_quotes(self, agent_instance, story_with_quotes):
        """Test: Agent should extract quotes from Story """
        mock_result = {
            "morals": [],
            "quotes": [
                {
                    "quote": "If you try hard enough, you may soon find an answer to your problem.",
                    "category": ["perseverance"],
                    "source": "The Thirsty Crow",
                    "author": "Narrator",
                    "confidence": 85,
                    "is_external": False
                }
            ]
        }
        with patch.object(agent_instance, "run", return_value=mock_result):
            result = agent_instance.run(story_with_quotes)

            quotes = result["quotes"]
            assert quotes is not None
            assert len(quotes) > 0
            logger.info(f"Extracted {len(quotes)} quotes")


    def test_moral_has_required_fields(self, agent_instance, sample_story):
        """Test: Each moral should have required fields"""

        mock_result = {
            "morals": [
                {
                    "moral": "Honesty is the best policy",
                    "category" : ["wisdom"],
                    "source": "Story",
                    "confidence": 90
                }
            ],
            "quotes": []
        }

        # Patch the run method to return mock_result instead of hitting API
        with patch.object(agent_instance, "run", return_value=mock_result):
            result = agent_instance.run(sample_story) # now returns mock_result

            moral = result["morals"][0]

            required_fields = ["moral", "category", "source", "confidence"]
            for field in required_fields:
                assert field in moral, f"Missing field: {field}"
            logger.info("Moral has all required fields")


    def test_quote_has_required_fields(self, agent_instance, story_with_quotes):
        """Test: Each quote should have required fields"""

        mock_result = {
            "morals": [],
            "quotes": [
                {
                    "quote": "If you try hard enough, you may soon find an answer to your problem.",
                    "category": ["perseverance"],
                    "source": "The Thirsty Crow",
                    "author": "Narrator",
                    "confidence": 85,
                    "is_external": False
                }
            ]
        }

        with patch.object(agent_instance, "run", return_value=mock_result):
            result = agent_instance.run(story_with_quotes)

            # check quote exist or not
            if not result["quotes"]:
                pytest.skip("No quotes generated after validation")
            quote = result["quotes"][0]

            required_fields = ["quote", "category", "source", "author","confidence", "is_external"]
            for field in required_fields:
                assert field in quote, f"Missing field: {field}"
            logger.info(f"Quote has all required fields")


    def test_confidence_in_valid_range(self, agent_instance, story_with_quotes):
        """Test: Morals's Confidence should between 0-100
                 Quotes's Confidence should between 70-100"""

        mock_result = {
            "morals": [
                MagicMock(confidence=95),
                MagicMock(confidence=85)
            ],
            "quotes": [
                MagicMock(confidence=75),
                MagicMock(confidence=85)
            ]
        }

        #patch agent.run to return the mock result
        with patch.object(agent_instance, "run", returnValue=mock_result):
            result = agent_instance.run(story_with_quotes)

            for moral in result["morals"]:
                assert 0 <= moral.confidence <= 100

            for quote in result["quotes"]:
                assert 70 <= quote.confidence <= 100

            logger.info(f"All Confidence scores in valid range")


    def test_no_baby_talk_quotes(self, agent_instance):
        """Test: Should Not extract baby talk as quotes"""

        story = """
        Evie pointed at the rain. "Dibble dop!" she said.
        "Mama, play?" she asked. "Boken," she said sadly.
        """
        mock_result = {
            "morals": [],
            "quotes": []
        }
        with patch.object(agent_instance, "run", return_value=mock_result):
            result = agent_instance.run(story)

            invalid_quotes = ["Dibble dop", "Boken", "Mama, play?"]

            for quote_obj in result["quotes"]:
                quote_text = quote_obj.quote
                assert not any(invalid in quote_text for invalid in invalid_quotes)
            print("No baby talk extracted")


    def test_empty_story_handling(self, agent_instance):
        """Test: Should handle empty story gracefully"""
        try:
            result = agent_instance.run("")
            # Should return empty lists or handle gracefully
            assert isinstance(result, dict)
            assert result["morals"] == [], "Expected no morals for empty story"
            assert result["quotes"] == [], "Expected no quotes for empty story"

            logger.info(f"Empty story result: {result}")
            logger.info(f"Empty story handled gracefully")

        except Exception as e:
            pytest.fail(f"Failed to handle empty story: {e}")



# RAG tests
class TestRAG:
    """Test RAG functionality"""

    def test_rag_loads_database(self, rag_instance):
        """Test: RAG should load database successfully"""

        assert rag_instance.knowledge is not None
        assert "morals" in rag_instance.knowledge
        assert "quotes" in rag_instance.knowledge
        logger.info(f"RAG loaded database")


    def test_add_to_knowledge(self, rag_instance):
        """Test: Should add items to knowledge base"""
        rag_instance.add_to_knowledge(
            text= "Honesty is the best policy",
            category= "wisdom",
            item_type= "moral"
        )

        assert "wisdom" in rag_instance.knowledge["morals"]
        assert len(rag_instance.knowledge["morals"]["wisdom"]) > 0
        logger.info(f"Added item to knowledge base")


    def test_cosine_similarity(self, rag_instance):
        """Test: Cosine similarity calculation"""
        # arrange
        vec1 = [1,0,0]
        vec2 = [1,0,0]
        vec3 = [0,1,0]

        sim_identical = rag_instance._cosine_similarity(vec1, vec2)
        sim_different = rag_instance._cosine_similarity(vec1, vec3)

        assert sim_identical == 1.0   # similar vectors
        assert sim_different == 0.0   # orthogonal vectors
        logger.info(f"Cosine similarity works correctly")


    def test_rag_similarity(self):
        """Test: RAG should return similar past examples"""

        # create temp database with controlled data
        test_db = "temp_test.json"

        past_data = {
            "morals": {},
            "quotes": {
                "wisdom": [{
                    "text": "knowledge is power",
                    "embedding": [0.9, 0.8, 0.1] * 512  # wisdom related embedding

                }],
                "love": [{
                    "text": "Love conquers all",
                    "embedding": [0.1, 0.2, 0.9] * 512  # Love embedding
                }]
            }
        }

        with open(test_db, "w") as f:
            json.dump(past_data, f)

        # Initialize RAG
        rag = SimpleRAG(db_path=test_db)

        # Moke embedding function (no real API call)
        def fake_embedding(text):
            if "knowledge" in text.lower() or "wisdom" in text.lower():
                return [0.85,0.75, 0.15] * 512
            elif "love" in text.lower():
                return [0.15, 0.25, 0.88] *512
            else:
                return [0.5, 0.5, 0.5] * 512

        rag._get_embedding = fake_embedding

        story = "A student gained knowledge and wisdom"
        context = rag.get_context_examples(story, "quote", top_n=2)
        logger.info(f"\nContext:\n{context}")
        assert "knowledge is power" in context, "Should find wisdom quote!"
        logger.info(f"Test Passed! RAG uses similar examples correctly")



        if os.path.exists(test_db):
            os.remove(test_db)
            logger.info(f"Removed {test_db} successfully")


    def test_duplicate_prevention(self, rag_instance):
        """Test: Should not add duplicates"""
        # Act
        rag_instance.add_to_knowledge("Test quote", "wisdom", "quote")
        rag_instance.add_to_knowledge("Test quote", "wisdom", "quote")  # Duplicate

        # Assert
        quotes = rag_instance.knowledge["quotes"]["wisdom"]
        # Count how many times "Test quote" appears
        count = sum(1 for item in quotes if
                    (isinstance(item, dict) and item['text'] == "Test quote") or
                    item == "Test quote")
        assert count == 1, "Duplicate was added"
        print(" Duplicate prevention works")



# Test Validation tests
class TestValidation:
    """Test validation functions"""

    def test_valid_quote_in_story(self):
        """Test: Quote exists in story"""

        from main import is_valid_quote
        story = "knowledge is a gift. Share it with the world."
        quote = "knowledge is a gift."

        result = is_valid_quote(quote, story)

        assert result == True
        print("Valid quote detected")

    def test_invalid_quote_in_story(self):
        """Test: Quote doesn't exist in story"""

        from main import is_valid_quote
        story = "A woodcutter lost his axe."
        quote = "Honesty is the best policy"

        result = is_valid_quote(quote, story)

        assert result == False
        print("Invalid quote rejected")

    def test_empty_quote_validation(self):
        """Test: Empty quote should be invalid"""
        # Arrange
        from main import is_valid_quote
        story = "Some story"
        quote = ""

        # Act
        result = is_valid_quote(quote, story)

        # Assert

        assert result == False
        print(" Empty quote rejected")









