import logging
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Dict, List
import pandas as pd

# Import pipeline components
from data_collection.fetch_reddit_data import RedditDataFetcher
from data_transformation.clean_data import TextCleaner
from data_transformation.normalize_fields import TextNormalizer
from sentiment_analysis.sentiment_vader import VaderSentimentAnalyzer
from sentiment_analysis.sentiment_transformers import TransformerSentimentAnalyzer

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentAnalysisPipeline:
    """
    A pipeline class that coordinates the collection, preprocessing, and sentiment
    analysis of Reddit content using multiple sentiment analysis approaches.
    """

    def __init__(self):
        """Initialize all components of the pipeline."""
        # Initialize data fetcher
        credentials_path = Path(__file__).parent / '/Users/tommya/Desktop/12-24_dev/redditlens/config.yaml'
        self.fetcher = RedditDataFetcher(credentials_path)

        # Initialize text preprocessing
        self.cleaner = TextCleaner(
            remove_urls=True,
            remove_emojis=True,
            remove_hashtags=True,
            remove_user_mentions=True,
            lowercase=True
        )

        self.normalizer = TextNormalizer(
            lemmatize=True,
            remove_stopwords=False,
            keep_important_stopwords=True
        )

        # Initialize sentiment analyzers
        self.vader_analyzer = VaderSentimentAnalyzer()
        self.transformer_analyzer = TransformerSentimentAnalyzer(batch_size=8)

    def analyze_subreddit_topic(
            self,
            keywords: List[str],
            subreddits: List[str],
            limit: int = 100,
            include_comments: bool = True,
            comment_limit: int = 50
    ) -> Dict:
        """
        Collect and analyze Reddit content about a specific topic.

        Args:
            keywords: List of keywords to search for
            subreddits: List of subreddits to search in
            limit: Maximum number of submissions to fetch per subreddit
            include_comments: Whether to analyze comments as well
            comment_limit: Maximum number of comments to fetch per submission

        Returns:
            Dictionary containing analysis results and metadata
        """
        logger.info(f"Starting analysis for keywords {keywords} in subreddits {subreddits}")

        # Fetch submissions
        submissions = self.fetcher.search_keywords(
            keywords=keywords,
            subreddits=subreddits,
            limit=limit
        )

        if not submissions:
            logger.warning("No submissions found matching criteria")
            return {"error": "No submissions found"}

        # Clean and normalize submissions
        logger.info("Preprocessing submissions...")
        cleaned_submissions = self.cleaner.clean_batch(submissions, 'submission')
        processed_submissions = self.normalizer.normalize_batch(cleaned_submissions, 'submission')

        # Analyze submissions with both methods
        logger.info("Analyzing submissions with VADER...")
        vader_results = self.vader_analyzer.analyze_batch(
            processed_submissions,
            'submission',
            use_normalized=True
        )

        logger.info("Analyzing submissions with Transformer model...")
        transformer_results = self.transformer_analyzer.analyze_batch(
            processed_submissions,
            'submission',
            use_normalized=True
        )

        # Process comments if requested
        comments_analysis = {}
        if include_comments:
            all_comments = []
            for submission in submissions:
                submission_comments = self.fetcher.fetch_submission_comments(
                    submission['id'],
                    limit=comment_limit
                )
                all_comments.extend(submission_comments)

            if all_comments:
                logger.info("Preprocessing comments...")
                cleaned_comments = self.cleaner.clean_batch(all_comments, 'comment')
                processed_comments = self.normalizer.normalize_batch(cleaned_comments, 'comment')

                logger.info("Analyzing comments with VADER...")
                vader_comment_results = self.vader_analyzer.analyze_batch(
                    processed_comments,
                    'comment',
                    use_normalized=True
                )

                logger.info("Analyzing comments with Transformer model...")
                transformer_comment_results = self.transformer_analyzer.analyze_batch(
                    processed_comments,
                    'comment',
                    use_normalized=True
                )

                comments_analysis = {
                    'vader': vader_comment_results,
                    'transformer': transformer_comment_results
                }

        # Compile results
        analysis_results = {
            'metadata': {
                'keywords': keywords,
                'subreddits': subreddits,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'submission_count': len(submissions),
                'comment_count': len(comments_analysis.get('vader', [])) if include_comments else 0
            },
            'submissions': {
                'vader': vader_results,
                'transformer': transformer_results
            },
            'comments': comments_analysis
        }

        return analysis_results

    def compare_sentiment_results(self, analysis_results: Dict) -> pd.DataFrame:
        """
        Create a comparison DataFrame of VADER and Transformer results.

        Args:
            analysis_results: Results dictionary from analyze_subreddit_topic

        Returns:
            DataFrame containing side-by-side sentiment comparisons
        """
        comparisons = []

        # Compare submission sentiments
        for v_sub, t_sub in zip(
                analysis_results['submissions']['vader'],
                analysis_results['submissions']['transformer']
        ):
            comparison = {
                'content_type': 'submission',
                'id': v_sub['id'],
                'text': v_sub.get('title', '') + ' ' + v_sub.get('text', ''),
                'vader_sentiment': v_sub['overall_sentiment']['sentiment'],
                'vader_compound': v_sub['overall_sentiment']['compound'],
                'transformer_sentiment': t_sub['overall_sentiment']['sentiment'],
                'transformer_compound': t_sub['overall_sentiment']['compound'],
                'agreement': v_sub['overall_sentiment']['sentiment'] ==
                             t_sub['overall_sentiment']['sentiment']
            }
            comparisons.append(comparison)

        # Compare comment sentiments if available
        if 'comments' in analysis_results and analysis_results['comments']:
            for v_com, t_com in zip(
                    analysis_results['comments']['vader'],
                    analysis_results['comments']['transformer']
            ):
                comparison = {
                    'content_type': 'comment',
                    'id': v_com['id'],
                    'text': v_com.get('text', ''),
                    'vader_sentiment': v_com['sentiment']['sentiment'],
                    'vader_compound': v_com['sentiment']['compound'],
                    'transformer_sentiment': t_com['sentiment']['sentiment'],
                    'transformer_compound': t_com['sentiment']['compound'],
                    'agreement': v_com['sentiment']['sentiment'] ==
                                 t_com['sentiment']['sentiment']
                }
                comparisons.append(comparison)

        return pd.DataFrame(comparisons)


if __name__ == "__main__":
    # Example usage
    pipeline = SentimentAnalysisPipeline()

    # Analyze discussions about AI
    results = pipeline.analyze_subreddit_topic(
        keywords=["artificial intelligence", "AI", "machine learning"],
        subreddits=["artificial", "MachineLearning", "technology"],
        limit=10,
        include_comments=True,
        comment_limit=20
    )

    # Compare results
    comparison_df = pipeline.compare_sentiment_results(results)

    # Print summary statistics
    print("\nSentiment Analysis Comparison:")
    print(f"Total items analyzed: {len(comparison_df)}")
    print(f"Agreement rate: {(comparison_df['agreement'].mean() * 100):.1f}%")

    # Show some example disagreements
    disagreements = comparison_df[~comparison_df['agreement']]
    if not disagreements.empty:
        print("\nExample disagreements between VADER and Transformer:")
        for _, row in disagreements.head().iterrows():
            print(f"\nContent type: {row['content_type']}")
            print(f"Text: {row['text'][:100]}...")
            print(f"VADER: {row['vader_sentiment']} (compound: {row['vader_compound']:.2f})")
            print(f"Transformer: {row['transformer_sentiment']} (compound: {row['transformer_compound']:.2f})")

    # Save results for further analysis
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    results_path = Path(f"analysis_results_{timestamp}.json")
    comparison_path = Path(f"sentiment_comparison_{timestamp}.csv")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    comparison_df.to_csv(comparison_path, index=False)

    print(f"\nResults saved to {results_path}")
    print(f"Comparison saved to {comparison_path}")
