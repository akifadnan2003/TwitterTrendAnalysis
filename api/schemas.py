from pydantic import BaseModel, Field
from typing import List

class CommentInput(BaseModel):
    text: str
   
    reach: int = Field(0, description="The estimated reach or follower count of the user's account.")

class SuggestionOutput(BaseModel):
    predicted_likes: float
    similar_comments: List[str]
    suggested_keywords: List[str]
