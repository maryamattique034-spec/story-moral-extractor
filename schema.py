from pydantic import BaseModel
from typing import Optional
# Schema

class MoralSchema(BaseModel):
    moral: str
    category: str
    source: str
    confidence: int

class QuoteSchema(BaseModel):
    quote: str
    category: str
    author: str
    confidence: int

class QuotesResponseSchema(BaseModel):
    response: list[QuoteSchema]

class ResponseSchema(BaseModel):
    response: list[MoralSchema]