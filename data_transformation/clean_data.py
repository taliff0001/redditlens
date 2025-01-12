import re
from typing import Dict, List, Union, Optional
import logging
from bs4 import BeautifulSoup
import unicodedata
import emoji

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    A class to clean and preprocess text data from Reddit submissions and comments.
    Handles common text cleaning tasks like removing URLs, special characters, and
    normalizing text format.
    """

    def __init__(
            self,
            remove_urls: bool = True,
            remove_emojis: bool = True,
            remove_hashtags: bool = True,
            remove_user_mentions: bool = True,
            lowercase: bool = True,
            normalize_whitespace: bool = True,
            remove_html: bool = True
    ):
        """
        Initialize the text cleaner with specified cleaning options.

        Args:
            remove_urls: Whether to remove URLs from text
            remove_emojis: Whether to remove emojis
            remove_hashtags: Whether to remove hashtags
            remove_user_mentions: Whether to remove user mentions (e.g., @user)
            lowercase: Whether to convert text to lowercase
            normalize_whitespace: Whether to normalize whitespace
            remove_html: Whether to remove HTML tags
        """
        self.remove_urls = remove_urls
        self.remove_emojis = remove_emojis
        self.remove_hashtags = remove_hashtags
        self.remove_user_mentions = remove_user_mentions
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        self.remove_html = remove_html

        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.whitespace_pattern = re.compile(r'\s+')

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string according to the configured options.

        Args:
            text: The text string to clean

        Returns:
            Cleaned text string
        """
        if not text:
            return ""

        try:
            # Remove HTML if enabled
            if self.remove_html:
                text = BeautifulSoup(text, 'html.parser').get_text()

            # Remove URLs if enabled
            if self.remove_urls:
                text = self.url_pattern.sub('', text)

            # Remove emojis if enabled
            if self.remove_emojis:
                text = emoji.replace_emoji(text, '')

            # Remove hashtags if enabled
            if self.remove_hashtags:
                text = self.hashtag_pattern.sub('', text)

            # Remove user mentions if enabled
            if self.remove_user_mentions:
                text = self.mention_pattern.sub('', text)

            # Convert to lowercase if enabled
            if self.lowercase:
                text = text.lower()

            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)

            # Normalize whitespace if enabled
            if self.normalize_whitespace:
                text = self.whitespace_pattern.sub(' ', text).strip()

            return text

        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            logger.error(f"Problematic text: {text[:100]}...")  # Log first 100 chars
            return text  # Return original text if cleaning fails

    def clean_submission(self, submission: Dict) -> Dict:
        """
        Clean text fields in a Reddit submission.

        Args:
            submission: Dictionary containing submission data

        Returns:
            Dictionary with cleaned text fields
        """
        cleaned_submission = submission.copy()

        try:
            # Clean title and text fields
            cleaned_submission['title'] = self.clean_text(submission['title'])
            cleaned_submission['text'] = self.clean_text(submission['text'])
            cleaned_submission['cleaned'] = True  # Mark as cleaned

            return cleaned_submission

        except Exception as e:
            logger.error(f"Error cleaning submission {submission.get('id', 'unknown')}: {str(e)}")
            return submission

    def clean_comment(self, comment: Dict) -> Dict:
        """
        Clean text fields in a Reddit comment.

        Args:
            comment: Dictionary containing comment data

        Returns:
            Dictionary with cleaned text fields
        """
        cleaned_comment = comment.copy()

        try:
            # Clean text field
            cleaned_comment['text'] = self.clean_text(comment['text'])
            cleaned_comment['cleaned'] = True  # Mark as cleaned

            return cleaned_comment

        except Exception as e:
            logger.error(f"Error cleaning comment {comment.get('id', 'unknown')}: {str(e)}")
            return comment

    def clean_batch(
            self,
            items: List[Dict],
            item_type: str = 'submission'
    ) -> List[Dict]:
        """
        Clean a batch of submissions or comments.

        Args:
            items: List of dictionaries containing submissions or comments
            item_type: Type of items ('submission' or 'comment')

        Returns:
            List of dictionaries with cleaned text fields
        """
        if item_type not in ['submission', 'comment']:
            raise ValueError("item_type must be either 'submission' or 'comment'")

        clean_method = self.clean_submission if item_type == 'submission' else self.clean_comment
        cleaned_items = []

        for item in items:
            try:
                cleaned_item = clean_method(item)
                cleaned_items.append(cleaned_item)
            except Exception as e:
                logger.error(f"Error in batch cleaning: {str(e)}")
                cleaned_items.append(item)  # Add original item if cleaning fails

        return cleaned_items


if __name__ == '__main__':
    # Example usage
    cleaner = TextCleaner(
        remove_urls=True,
        remove_emojis=True,
        remove_hashtags=True,
        remove_user_mentions=True,
        lowercase=True
    )

    # Example text with various elements to clean
    sample_text = """
    Check out this cool link! https://example.com
    @user mentioned something about #DataScience 🚀
    Here's some <b>HTML</b> content.
    """

    cleaned_text = cleaner.clean_text(sample_text)
    print(f"Original text:\n{sample_text}\n")
    print(f"Cleaned text:\n{cleaned_text}")