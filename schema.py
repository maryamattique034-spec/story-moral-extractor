from pydantic import BaseModel
from typing import List
# Schema

class MoralSchema(BaseModel):
    moral: str
    category: List[str]
    source: str
    confidence: int

class QuoteSchema(BaseModel):
    quote: str
    category: List[str]
    source : str
    author: str
    confidence: int
    is_external : bool

class QuotesResponseSchema(BaseModel):
    response: list[QuoteSchema]

class ResponseSchema(BaseModel):
    response: list[MoralSchema]