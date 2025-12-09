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
    def __init__(self, open_api_key: str):
        self._client = OpenAI(api_key=open_api_key, )
        self.model = "gpt-4o-mini-2024-07-18"
        self.we_did_not_specify_stop_tokens = True

    def run(self, input: str):
        """run the agent"""
        try:
            moral_prompt  = self.prompt("prompt_files/prompt.txt", input)
            quote_prompt  = self.prompt("prompt_files/quote_prompt.txt", input)
            moral_output = self.call(moral_prompt, ResponseSchema)
            quote_output = self.call(quote_prompt, QuotesResponseSchema)
            sanitized_moral_output = self.sanitize_output(moral_output)
            sanitized_quote_output = self.sanitize_output(quote_output)
            sanitized_output = {"morals":sanitized_moral_output.response, "quotes": sanitized_quote_output.response}
            return sanitized_output
        except AgentException as e:
            raise e
        except Exception as exc:
            raise AgentException(f"Error: {str(exc)}")
    
    def prompt(self,file_path: str, story: str):
        """prompt the agent"""
        with open(file_path, "r") as file:
            return file.read().format(story=story)


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
