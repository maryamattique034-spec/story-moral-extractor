from openai import OpenAI
from pydantic import BaseModel
from schema import MoralSchema, ResponseSchema, QuotesResponseSchema
import os   
import json


# Exception Handler
class AgentException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


# Agent

class Agent:
    def __init__(self, open_api_key: str, rag = None):
        self._client = OpenAI(api_key=open_api_key )
        self.model = "gpt-4o-mini-2024-07-18"
        self.rag = rag # Store rag instance
        self.we_did_not_specify_stop_tokens = True


    def is_baby_talk(self, text:str) -> bool:
        """
        Detects baby-talk / nonsense quotes
        """
        text = text.strip('"\'')  # Remove Quotes too
        words= text.split()

        # Empty or single word
        if len(words) <= 1:
            return True

        baby_patterns = [
            "dibble", "dop", "boken", "mama", "dada",
            "tweet", "woof", "meow", "moo"
        ]

        text_lower = text.lower()
        if any(pattern in text_lower for pattern in baby_patterns):
            return True

        # Mostly short nonsense words (2 char or less)
        short_words = [w for w in words if len(w) <= 2]
        if len(short_words)/ len(words) > 0.5:      # More than 50% tiny words
            return True

        return False


    def run(self, input: str):
        """run the agent with embedding-based RAG"""

        # Guard : empty or meaningless input
        if not input or not input.strip():
            return {
                "morals": [],
                "quotes": []
            }
        try:
            # Get past examples/context from RAG (RETRIEVAL step)
            moral_context = ""
            quote_context = ""

            if self.rag:
                moral_context = self.rag.get_context_examples(input,"moral", top_n=10)
                quote_context = self.rag.get_context_examples(input, "quote", top_n=10)

            # Pass context to prompts (AUGMENTATION step)
            moral_prompt  = self.prompt("prompt_files/prompt.txt", input, moral_context)
            quote_prompt  = self.prompt("prompt_files/quote_prompt.txt", input, quote_context)

            # Generation step (AI generates with relevant context)
            moral_output = self.call(moral_prompt, ResponseSchema)
            quote_output = self.call(quote_prompt, QuotesResponseSchema)

            sanitized_moral_output = self.sanitize_output(moral_output)
            sanitized_quote_output = self.sanitize_output(quote_output)

            # Filter baby-talk quotes
            filtered_quotes = [q for q in sanitized_quote_output.response
                               if not self.is_baby_talk(q.quote)
            ]

            sanitized_output = {
                "morals":sanitized_moral_output.response,
                "quotes": filtered_quotes
            }
            return sanitized_output

        except AgentException as e:
            raise e
        except Exception as exc:
            raise AgentException(f"Error: {str(exc)}")


    def prompt(self,file_path: str, story: str, past_context: str = ""):
        """prompt the agent with past context parameter"""
        with open(file_path, "r") as file:
            template = file.read()
            return template.format(story=story, past_context=past_context)


    def sanitize_output(self, response: BaseModel):
        # Check if the conversation was too long for the context window, resulting in incomplete JSON 
        if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
            # your code should handle this error case
            raise AgentException("The conversation was too long for the context window, resulting in incomplete JSON")

        # Check if the OpenAI safety system refused the request and generated a refusal instead
        if response.output[0].content[0].type == "refusal":
            # your code should handle this error case
            # In this case, the .content field will contain the explanation (if any) that the model generated for why it is refusing
            raise AgentException(response.output[0].content[0]["refusal"])

        # Check if the model's output included restricted content, so the generation of JSON was halted and may be partial
        if response.status == "incomplete" and response.incomplete_details.reason == "content_filter":
            # your code should handle this error case
            raise AgentException("The model's output included restricted content, so the generation of JSON was halted and may be partial")

        if response.status == "completed":
            # In this case the model has either successfully finished generating the JSON object according to your schema, or the model generated one of the tokens you provided as a "stop token"

            if self.we_did_not_specify_stop_tokens:
                # If you didn't specify any stop tokens, then the generation is complete and the content key will contain the serialized JSON object
                # This will parse successfully and should now contain  "{"winner": "Los Angeles Dodgers"}"
                return response.output_parsed
            else:
                # Check if the response.output_text ends with one of your stop tokens and handle appropriately
                raise AgentException("The response.output_text ends with one of your stop tokens")
        return response

    def call(self, prompt: str, output_parser: BaseModel):
        """extract the morals from the text"""

        response = self._client.responses.parse(
        model=self.model,
        input=[
            {"role":"system", "content": prompt},
            
        ],
        text_format=output_parser,
        temperature=0.5,
        )
        
        return response
