import re
import string

def clean_tweet_text(text):
    """
    Applies text cleaning for real-world tweets:
    - Lowercase
    - Removes URLs
    - Removes user mentions (@username)
    - Removes hashtags symbols (#) but keeps the text
    - Removes punctuation
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = text.replace('#', '')
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text.strip()
