import spacy
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    A class to normalize text through lemmatization and other linguistic transformations.
    Uses spaCy for advanced natural language processing capabilities.
    """

    def __init__(
            self,
            lemmatize: bool = True,
            remove_stopwords: bool = False,
            keep_important_stopwords: bool = True,
            batch_size: int = 1000
    ):
        """
        Initialize the text normalizer with specified options.

        Args:
            lemmatize: Whether to perform lemmatization
            remove_stopwords: Whether to remove stopwords
            keep_important_stopwords: Whether to keep stopwords that may be important
                                    for sentiment analysis (e.g., 'not', 'very', 'no')
            batch_size: Size of batches for processing large amounts of text
        """
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.keep_important_stopwords = keep_important_stopwords
        self.batch_size = batch_size

        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')

            # Configure pipeline for efficiency
            # Only keep components we need for lemmatization
            disabled_pipes = []
            for pipe in self.nlp.pipe_names:
                if pipe not in ['tok2vec', 'lemmatizer', 'attribute_ruler']:
                    disabled_pipes.append(pipe)
            logger.info(f"Available pipeline components: {self.nlp.pipe_names}")
            logger.info(f"Disabling unnecessary components: {disabled_pipes}")
            self.nlp.disable_pipes(disabled_pipes)
            logger.info(f"Active pipeline components: {[name for name, pipe in self.nlp.pipeline]}")

            # Define sentiment-important stopwords
            self.important_stopwords = {
                'no', 'not', 'very', 'never', 'none', 'nothing',
                'few', 'more', 'most', 'always', 'too', 'only',
                'just', 'any', 'all', 'but'
            }

        except OSError:
            logger.error("Required spaCy model 'en_core_web_sm' not found. Installing...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

    def normalize_text(self, text: str) -> str:
        """
        Normalize a single text string through lemmatization and other transformations.

        Args:
            text: The text string to normalize

        Returns:
            Normalized text string
        """
        if not text:
            return ""

        try:
            doc = self.nlp(text)

            normalized_tokens = []
            for token in doc:
                # Skip empty tokens and whitespace
                if token.is_space or not token.text.strip():
                    continue

                # Handle stopwords
                if self.remove_stopwords and token.is_stop:
                    if self.keep_important_stopwords and token.lower_ in self.important_stopwords:
                        # Keep important stopwords if configured
                        normalized_tokens.append(token.text)
                    continue

                # Apply lemmatization if enabled
                if self.lemmatize:
                    # Handle special cases where lemmatization might not be desired
                    if token.pos_ in ['PROPN', 'NUM']:  # Proper nouns and numbers
                        normalized_tokens.append(token.text)
                    else:
                        normalized_tokens.append(token.lemma_)
                else:
                    normalized_tokens.append(token.text)

            return ' '.join(normalized_tokens)

        except Exception as e:
            logger.error(f"Error normalizing text: {str(e)}")
            logger.error(f"Problematic text: {text[:100]}...")  # Log first 100 chars
            return text  # Return original text if normalization fails

    def normalize_submission(self, submission: Dict) -> Dict:
        """
        Normalize text fields in a Reddit submission.

        Args:
            submission: Dictionary containing submission data

        Returns:
            Dictionary with normalized text fields
        """
        normalized_submission = submission.copy()

        try:
            # Normalize title and text fields
            normalized_submission['title_normalized'] = self.normalize_text(submission['title'])
            normalized_submission['text_normalized'] = self.normalize_text(submission['text'])
            normalized_submission['normalized'] = True  # Mark as normalized

            # Keep original fields
            normalized_submission['title_original'] = submission['title']
            normalized_submission['text_original'] = submission['text']

            return normalized_submission

        except Exception as e:
            logger.error(f"Error normalizing submission {submission.get('id', 'unknown')}: {str(e)}")
            return submission

    def normalize_comment(self, comment: Dict) -> Dict:
        """
        Normalize text fields in a Reddit comment.

        Args:
            comment: Dictionary containing comment data

        Returns:
            Dictionary with normalized text fields
        """
        normalized_comment = comment.copy()

        try:
            # Normalize text field
            normalized_comment['text_normalized'] = self.normalize_text(comment['text'])
            normalized_comment['normalized'] = True  # Mark as normalized

            # Keep original text
            normalized_comment['text_original'] = comment['text']

            return normalized_comment

        except Exception as e:
            logger.error(f"Error normalizing comment {comment.get('id', 'unknown')}: {str(e)}")
            return comment

    def normalize_batch(
            self,
            items: List[Dict],
            item_type: str = 'submission'
    ) -> List[Dict]:
        """
        Normalize a batch of submissions or comments.

        Args:
            items: List of dictionaries containing submissions or comments
            item_type: Type of items ('submission' or 'comment')

        Returns:
            List of dictionaries with normalized text fields
        """
        if item_type not in ['submission', 'comment']:
            raise ValueError("item_type must be either 'submission' or 'comment'")

        normalize_method = self.normalize_submission if item_type == 'submission' else self.normalize_comment
        normalized_items = []

        # Process in batches for better performance
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]

            for item in batch:
                try:
                    normalized_item = normalize_method(item)
                    normalized_items.append(normalized_item)
                except Exception as e:
                    logger.error(f"Error in batch normalization: {str(e)}")
                    normalized_items.append(item)  # Add original item if normalization fails

        return normalized_items


if __name__ == '__main__':
    # Example usage
    normalizer = TextNormalizer(
        lemmatize=True,
        remove_stopwords=True,
        keep_important_stopwords=True
    )

    # Example text with various forms needing normalization
    sample_text = """
    The cats are running quickly and the dogs were barking loudly.
    I'm not very happy about this situation, but I'll try to remain positive.
    """

    normalized_text = normalizer.normalize_text(sample_text)
    print(f"Original text:\n{sample_text}\n")
    print(f"Normalized text:\n{normalized_text}")
