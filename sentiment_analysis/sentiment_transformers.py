from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timezone
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransformerSentimentAnalyzer:
    """
    A class for sentiment analysis using pre-trained transformer models.
    Uses the RoBERTa model fine-tuned on social media text for sentiment analysis.
    The model provides more nuanced sentiment understanding compared to lexicon-based
    approaches by considering context and modern language patterns.
    """

    def __init__(
            self,
            model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
            device: Optional[str] = None,
            batch_size: int = 8,
            max_length: int = 512
    ):
        """
        Initialize the transformer-based sentiment analyzer.

        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
            batch_size: Number of texts to process at once
            max_length: Maximum length of input text (longer texts will be truncated)
        """
        self.batch_size = batch_size
        self.max_length = max_length

        # Set up device for computation
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Map model outputs to sentiment labels
            self.id2label = self.model.config.id2label

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def _prepare_text_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of texts for model inference.

        Args:
            texts: List of text strings to process

        Returns:
            Dictionary of tokenized inputs
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

    def _process_model_output(self, outputs) -> List[Dict[str, Union[float, str]]]:
        """
        Process raw model outputs into sentiment scores and labels.

        Args:
            outputs: Model output tensors

        Returns:
            List of dictionaries containing sentiment scores and labels
        """
        # Get probabilities using softmax
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        results = []
        for prob in probs:
            # Convert to numpy for easier handling
            prob_dict = {self.id2label[i]: float(p) for i, p in enumerate(prob)}

            # Get the predicted label
            pred_label = self.id2label[prob.argmax().item()]

            # Calculate compound score (-1 to 1) for compatibility with VADER
            if 'negative' in prob_dict and 'positive' in prob_dict:
                compound = prob_dict['positive'] - prob_dict['negative']
            else:
                compound = 0.0

            results.append({
                **prob_dict,
                'sentiment': pred_label,
                'compound': compound
            })

        return results

    def analyze_texts(self, texts: List[str]) -> List[Dict[str, Union[float, str]]]:
        """
        Analyze sentiment for a list of texts.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of dictionaries containing sentiment scores and labels
        """
        if not texts:
            return []

        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            try:
                # Prepare batch
                inputs = self._prepare_text_batch(batch_texts)

                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Process outputs
                batch_results = self._process_model_output(outputs)
                results.extend(batch_results)

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                # Add neutral sentiment for failed texts
                results.extend([{
                    'error': str(e),
                    'sentiment': 'neutral',
                    'compound': 0.0
                }] * len(batch_texts))

        return results

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

            # Analyze both title and text
            results = self.analyze_texts([submission[title_field], submission[text_field]])

            if len(results) == 2:
                title_sentiment, text_sentiment = results

                analyzed_submission['title_sentiment'] = title_sentiment
                analyzed_submission['text_sentiment'] = text_sentiment

                # Calculate overall sentiment
                analyzed_submission['overall_sentiment'] = self._calculate_overall_sentiment(
                    title_sentiment,
                    text_sentiment,
                    0.3  # Title weight
                )

                # Add analysis metadata
                analyzed_submission['sentiment_analysis'] = {
                    'method': 'transformer',
                    'model': self.model.config.name_or_path,
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
            results = self.analyze_texts([comment[text_field]])

            if results:
                analyzed_comment['sentiment'] = results[0]

                # Add analysis metadata
                analyzed_comment['sentiment_analysis'] = {
                    'method': 'transformer',
                    'model': self.model.config.name_or_path,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
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
            Combined sentiment scores and label
        """
        text_weight = 1 - title_weight

        # Combine probabilities for each sentiment class
        combined_scores = {}
        for label in self.id2label.values():
            if label in title_sentiment and label in text_sentiment:
                combined_scores[label] = (
                        title_weight * title_sentiment[label] +
                        text_weight * text_sentiment[label]
                )

        # Calculate compound score
        compound = (
                title_weight * title_sentiment['compound'] +
                text_weight * text_sentiment['compound']
        )

        # Get overall sentiment label
        sentiment = max(combined_scores.items(), key=lambda x: x[1])[0]

        return {
            **combined_scores,
            'compound': compound,
            'sentiment': sentiment
        }

    def analyze_batch(
            self,
            items: List[Dict],
            item_type: str = 'submission',
            use_normalized: bool = True
    ) -> List[Dict]:
        """
        Analyze sentiment for a batch of submissions or comments.

        Args:
            items: List of dictionaries containing submissions or comments
            item_type: Type of items ('submission' or 'comment')
            use_normalized: Whether to use normalized text if available

        Returns:
            List of items with added sentiment analysis
        """
        if item_type not in ['submission', 'comment']:
            raise ValueError("item_type must be either 'submission' or 'comment'")

        analyze_method = self.analyze_submission if item_type == 'submission' else self.analyze_comment
        analyzed_items = []

        # Process items with progress bar
        for item in tqdm(items, desc=f"Analyzing {item_type}s"):
            try:
                analyzed_item = analyze_method(item, use_normalized)
                analyzed_items.append(analyzed_item)
            except Exception as e:
                logger.error(f"Error in batch analysis: {str(e)}")
                analyzed_items.append(item)

        return analyzed_items


if __name__ == '__main__':
    # Example usage
    analyzer = TransformerSentimentAnalyzer()

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

    print("\nOriginal Text Analysis:")
    print(f"Title Sentiment: {results_original['title_sentiment']}")
    print(f"Text Sentiment: {results_original['text_sentiment']}")
    print(f"Overall Sentiment: {results_original['overall_sentiment']}")

    print("\nNormalized Text Analysis:")
    print(f"Title Sentiment: {results_normalized['title_sentiment']}")
    print(f"Text Sentiment: {results_normalized['text_sentiment']}")
    print(f"Overall Sentiment: {results_normalized['overall_sentiment']}")