from typing import Dict, List, Optional, Union
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VaderSentimentAnalyzer:
    """
    A class to perform sentiment analysis using VADER, specifically optimized for
    social media content. VADER is particularly good at handling:
    - Social media conventions (e.g., emoticons, slang)
    - Punctuation emphasis (!!!)
    - Word-shape emphasis (ALL CAPS)
    - Degree modifiers (e.g., 'very', 'somewhat')
    - Conjunctions (e.g., 'but', 'however')
    - Preceding trigrams (e.g., 'could have been')
    """

    def __init__(self, compound_threshold: float = 0.05):
        """
        Initialize the VADER sentiment analyzer.

        Args:
            compound_threshold: The threshold for neutral sentiment classification
                              (-threshold to +threshold is considered neutral)
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.compound_threshold = compound_threshold

    def analyze_text(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment of a single text string.

        Args:
            text: The text to analyze

        Returns:
            Dictionary containing sentiment scores and classification:
            - neg: Negative score (0 to 1)
            - neu: Neutral score (0 to 1)
            - pos: Positive score (0 to 1)
            - compound: Normalized compound score (-1 to 1)
            - sentiment: Categorical sentiment (positive/negative/neutral)
        """
        if not text:
            return {
                'neg': 0.0,
                'neu': 1.0,
                'pos': 0.0,
                'compound': 0.0,
                'sentiment': 'neutral'
            }

        try:
            # Get the sentiment scores
            scores = self.analyzer.polarity_scores(text)

            # Add categorical sentiment based on compound score
            scores['sentiment'] = self._classify_sentiment(scores['compound'])

            return scores

        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            logger.error(f"Problematic text: {text[:100]}...")  # Log first 100 chars
            return {
                'neg': 0.0,
                'neu': 1.0,
                'pos': 0.0,
                'compound': 0.0,
                'sentiment': 'neutral',
                'error': str(e)
            }

    def _classify_sentiment(self, compound_score: float) -> str:
        """
        Classify the compound sentiment score into a category.

        Args:
            compound_score: The compound sentiment score (-1 to 1)

        Returns:
            Sentiment category: 'positive', 'negative', or 'neutral'
        """
        if compound_score >= self.compound_threshold:
            return 'positive'
        elif compound_score <= -self.compound_threshold:
            return 'negative'
        else:
            return 'neutral'

    def analyze_submission(
            self,
            submission: Dict,
            use_normalized: bool = True
    ) -> Dict:
        """
        Analyze sentiment of a Reddit submission.

        Args:
            submission: Dictionary containing submission data
            use_normalized: Whether to use normalized text if available

        Returns:
            Submission dictionary with added sentiment analysis
        """
        analyzed_submission = submission.copy()

        try:
            # Determine which text fields to use
            title_field = 'title_normalized' if use_normalized and 'title_normalized' in submission else 'title'
            text_field = 'text_normalized' if use_normalized and 'text_normalized' in submission else 'text'

            # Analyze title sentiment
            title_sentiment = self.analyze_text(submission[title_field])
            analyzed_submission['title_sentiment'] = title_sentiment

            # Analyze text sentiment
            text_sentiment = self.analyze_text(submission[text_field])
            analyzed_submission['text_sentiment'] = text_sentiment

            # Calculate overall sentiment
            analyzed_submission['overall_sentiment'] = self._calculate_overall_sentiment(
                title_sentiment,
                text_sentiment,
                0.3  # Title weight (text gets 0.7)
            )

            # Add analysis metadata
            analyzed_submission['sentiment_analysis'] = {
                'method': 'vader',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'used_normalized_text': use_normalized
            }

            return analyzed_submission

        except Exception as e:
            logger.error(f"Error analyzing submission {submission.get('id', 'unknown')}: {str(e)}")
            return submission

    def analyze_comment(
            self,
            comment: Dict,
            use_normalized: bool = True
    ) -> Dict:
        """
        Analyze sentiment of a Reddit comment.

        Args:
            comment: Dictionary containing comment data
            use_normalized: Whether to use normalized text if available

        Returns:
            Comment dictionary with added sentiment analysis
        """
        analyzed_comment = comment.copy()

        try:
            # Determine which text field to use
            text_field = 'text_normalized' if use_normalized and 'text_normalized' in comment else 'text'

            # Analyze sentiment
            sentiment = self.analyze_text(comment[text_field])
            analyzed_comment['sentiment'] = sentiment

            # Add analysis metadata
            analyzed_comment['sentiment_analysis'] = {
                'method': 'vader',
                'timestamp': datetime.utcnow().isoformat(),
                'used_normalized_text': use_normalized
            }

            return analyzed_comment

        except Exception as e:
            logger.error(f"Error analyzing comment {comment.get('id', 'unknown')}: {str(e)}")
            return comment

    def _calculate_overall_sentiment(
            self,
            title_sentiment: Dict[str, Union[float, str]],
            text_sentiment: Dict[str, Union[float, str]],
            title_weight: float = 0.3
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate overall sentiment by combining title and text sentiment.

        Args:
            title_sentiment: Sentiment scores for title
            text_sentiment: Sentiment scores for text
            title_weight: Weight to give to title sentiment (0 to 1)

        Returns:
            Combined sentiment scores
        """
        text_weight = 1 - title_weight

        # Calculate weighted average for each score
        overall = {
            'neg': title_weight * title_sentiment['neg'] + text_weight * text_sentiment['neg'],
            'neu': title_weight * title_sentiment['neu'] + text_weight * text_sentiment['neu'],
            'pos': title_weight * title_sentiment['pos'] + text_weight * text_sentiment['pos'],
            'compound': title_weight * title_sentiment['compound'] + text_weight * text_sentiment['compound']
        }

        # Add categorical sentiment based on compound score
        overall['sentiment'] = self._classify_sentiment(overall['compound'])

        return overall

    def analyze_batch(
            self,
            items: List[Dict],
            item_type: str = 'submission',
            use_normalized: bool = True,
            batch_size: int = 1000
    ) -> List[Dict]:
        """
        Analyze sentiment for a batch of submissions or comments.

        Args:
            items: List of dictionaries containing submissions or comments
            item_type: Type of items ('submission' or 'comment')
            use_normalized: Whether to use normalized text if available
            batch_size: Size of batches for processing

        Returns:
            List of items with added sentiment analysis
        """
        if item_type not in ['submission', 'comment']:
            raise ValueError("item_type must be either 'submission' or 'comment'")

        analyze_method = self.analyze_submission if item_type == 'submission' else self.analyze_comment
        analyzed_items = []

        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_size = len(batch)

            logger.info(f"Processing batch {i // batch_size + 1}, size: {batch_size}")

            for item in batch:
                try:
                    analyzed_item = analyze_method(item, use_normalized)
                    analyzed_items.append(analyzed_item)
                except Exception as e:
                    logger.error(f"Error in batch analysis: {str(e)}")
                    analyzed_items.append(item)

        return analyzed_items


if __name__ == '__main__':
    # Example usage
    analyzer = VaderSentimentAnalyzer()

    # Example Reddit submission
    submission = {
        "id": "abc123",
        "title": "This new feature is amazing and really helpful!",
        "text": "I was skeptical at first, but after trying it out, I'm truly impressed. "
                "The developers did a fantastic job.",
        "title_normalized": "new feature be amazing and really helpful",
        "text_normalized": "be skeptical at first but after try it out be truly impress "
                           "developer do fantastic job"
    }

    # Analyze with both original and normalized text
    results_original = analyzer.analyze_submission(submission, use_normalized=False)
    results_normalized = analyzer.analyze_submission(submission, use_normalized=True)

    print("Original Text Analysis:")
    print(f"Title Sentiment: {results_original['title_sentiment']}")
    print(f"Text Sentiment: {results_original['text_sentiment']}")
    print(f"Overall Sentiment: {results_original['overall_sentiment']}")

    print("\nNormalized Text Analysis:")
    print(f"Title Sentiment: {results_normalized['title_sentiment']}")
    print(f"Text Sentiment: {results_normalized['text_sentiment']}")
    print(f"Overall Sentiment: {results_normalized['overall_sentiment']}")