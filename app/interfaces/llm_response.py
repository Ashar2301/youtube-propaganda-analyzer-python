from pydantic import BaseModel

class LLMResponse(BaseModel):
    explanation: str
    tags: list[str]