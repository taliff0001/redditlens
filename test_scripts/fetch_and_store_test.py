import logging
from pathlib import Path
import unittest

from sqlalchemy import select

from data_collection.fetch_reddit_data import RedditDataFetcher
from data_storage.database_handler import DatabaseHandler, RedditSubmissionNew, RedditCommentNew
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define test parameters (consider moving these to a config file later)
TEST_KEYWORDS = ["climate change"]
TEST_SUBREDDITS = ["science", "environment"]
TEST_TIME_FILTER = "month"
TEST_LIMIT = 5
TEST_COMMENT_LIMIT = 3

# Construct the path to config.yaml (consider making this more robust)
script_dir = Path(__file__).parent
config_path = script_dir.parent / 'config.yaml'

class TestFetchAndStore(unittest.TestCase):
    """
    Test suite for fetching and storing Reddit data, mirroring the structure
    of SentimentAnalysisPipeline in sentiment_analysis_test.py.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up resources for all test methods, similar to SentimentAnalysisPipeline.
        """
        try:
            cls.fetcher = RedditDataFetcher(config_path)
            cls.db_handler = DatabaseHandler(config_path)
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            raise

    def fetch_and_store(
        self,
        keywords: List[str],
        subreddits: List[str],
        limit: int = TEST_LIMIT,
        time_filter: str = TEST_TIME_FILTER,
        include_comments: bool = True,
        comment_limit: int = TEST_COMMENT_LIMIT
    ) -> Dict:
        """
        Fetches and stores Reddit submissions and optionally comments,
        similar to analyze_subreddit_topic in SentimentAnalysisPipeline.

        Args:
            keywords: List of keywords to search for.
            subreddits: List of subreddits to search in.
            limit: Maximum number of submissions to fetch per subreddit.
            time_filter: Time filter for the search (e.g., "day", "week", "month").
            include_comments: Whether to fetch and store comments.
            comment_limit: Maximum number of comments to fetch per submission.

        Returns:
            A dictionary containing information about the stored data.
        """
        logger.info(f"Fetching submissions for keywords: {keywords} in subreddits: {subreddits}")
        submissions = self.fetcher.search_keywords(
            keywords=keywords,
            subreddits=subreddits,
            time_filter=time_filter,
            limit=limit
        )

        if not submissions:
            logger.warning("No submissions found matching criteria.")
            return {"submissions_stored": 0, "comments_stored": 0}

        # Log fetched submissions
        logger.info(f"Fetched {len(submissions)} submissions:")
        for submission in submissions:
            logger.info(f"  - Submission ID: {submission['id']}, Title: {submission['title']}")

        # Store submissions
        submission_store_result = self.db_handler.store_batch(submissions, item_type='submission')
        logger.info(
            f"Successfully stored {submission_store_result['success']} "
            f"out of {len(submissions)} submissions."
        )

        comments_stored = 0
        if include_comments:
            logger.info("Fetching comments for submissions...")
            for submission in submissions:
                submission_id = submission['id']
                comments = self.fetcher.fetch_submission_comments(
                    submission_id=submission_id,
                    limit=comment_limit
                )

                # Log fetched comments
                logger.info(f"  - Fetched {len(comments)} comments for submission {submission_id}:")
                for comment in comments:
                    logger.info(f"    - Comment ID: {comment['id']}, Text: {comment['text'][:50]}...")

                # Add submission_id to each comment for database storage
                for comment in comments:
                    comment['submission_id'] = submission_id

                comment_store_result = self.db_handler.store_batch(comments, item_type='comment')
                comments_stored += comment_store_result['success']
                logger.info(
                    f"    - Successfully stored {comment_store_result['success']} "
                    f"out of {len(comments)} comments for submission {submission_id}."
                )

        return {
            "submissions_stored": submission_store_result['success'],
            "comments_stored": comments_stored
        }

    def test_fetch_and_store_submissions(self):
        """
        Test fetching and storing submissions using the fetch_and_store method.
        """
        logger.info("Testing fetching and storing submissions...")

        # TEMPORARY: Call _search_submissions directly
        query = ' OR '.join(f'"{keyword}"' for keyword in TEST_KEYWORDS)
        submissions = self.fetcher._search_submissions(
            query=query,
            subreddit=TEST_SUBREDDITS[0],  # Use just the first subreddit for simplicity
            time_filter=TEST_TIME_FILTER,
            limit=TEST_LIMIT,
            sort="relevance"  # Use a specific sort method
        )

        # Call fetch_and_store to store submissions (and optionally comments)
        result = self.fetch_and_store(
            keywords=TEST_KEYWORDS,
            subreddits=TEST_SUBREDDITS,
            limit=TEST_LIMIT,
            time_filter=TEST_TIME_FILTER,
            include_comments=True,  # Now including comments
            comment_limit=TEST_COMMENT_LIMIT
        )

        # Check that submissions were stored
        self.assertGreaterEqual(result['submissions_stored'], 1, "No submissions were stored.")

        # If comments were fetched, check that they were stored as well
        if result['comments_stored'] > 0:
            self.assertGreaterEqual(result['comments_stored'], 1, "No comments were stored.")

    def test_fetch_and_store_comments(self):

        """
        Test fetching and storing comments using the fetch_and_store method.
        """
        logger.info("Testing fetching and storing comments...")
        result = self.fetch_and_store(
            keywords=TEST_KEYWORDS,
            subreddits=TEST_SUBREDDITS,
            limit=TEST_LIMIT,
            time_filter=TEST_TIME_FILTER,
            include_comments=True,
            comment_limit=TEST_COMMENT_LIMIT
        )
        self.assertGreaterEqual(result['submissions_stored'], 1, "No submissions were stored.")
        self.assertGreaterEqual(result['comments_stored'], 1, "No comments were stored.")

    def test_edge_cases(self):
        """
        Test various edge cases, such as empty keywords, invalid subreddits, etc.
        """
        logger.info("Testing edge cases...")

        # Test with empty keywords
        result = self.fetch_and_store(
            keywords=[],
            subreddits=TEST_SUBREDDITS,
            limit=TEST_LIMIT,
            time_filter=TEST_TIME_FILTER
        )
        self.assertEqual(result['submissions_stored'], 0, "Submissions were stored with empty keywords.")
        self.assertEqual(result['comments_stored'], 0, "Comments were stored with empty keywords.")

        # Test with an invalid subreddit
        result = self.fetch_and_store(
            keywords=TEST_KEYWORDS,
            subreddits=["this_is_an_invalid_subreddit"],
            limit=TEST_LIMIT,
            time_filter=TEST_TIME_FILTER
        )
        self.assertEqual(result['submissions_stored'], 0, "Submissions were stored with an invalid subreddit.")
        self.assertEqual(result['comments_stored'], 0, "Comments were stored with an invalid subreddit.")

    def test_database_state(self):
        """
        Test the state of the database after fetch and store operations.
        """
        logger.info("Testing database state after fetch and store operations...")

        # Fetch and store some data
        result = self.fetch_and_store(
            keywords=TEST_KEYWORDS,
            subreddits=TEST_SUBREDDITS,
            limit=TEST_LIMIT,
            time_filter=TEST_TIME_FILTER,
            include_comments=True,
            comment_limit=TEST_COMMENT_LIMIT
        )
        submissions_stored = result['submissions_stored']
        comments_stored = result['comments_stored']

        # Check if submissions were stored
        if submissions_stored > 0:
            # Get a stored submission ID (you might need to adjust this logic)
            submissions = self.fetcher.search_keywords(
                keywords=TEST_KEYWORDS,
                subreddits=TEST_SUBREDDITS,
                limit=1
            )
            if submissions:
                submission_id = submissions[0]['id']

                # Verify the submission exists in the database
                # Verify the submission exists in the database using the new table
                with self.db_handler.session_scope() as session:
                    submission = session.get(RedditSubmissionNew, submission_id)
                    self.assertIsNotNone(submission, f"Submission {submission_id} not found in database.")

                    # Verify comments were stored if applicable using the new table
                    if comments_stored > 0:
                        stmt = select(RedditCommentNew).where(
                            RedditCommentNew.submission_id == submission_id
                        )
                        result = session.execute(stmt)
                        comments = result.scalars().all()
                        self.assertGreaterEqual(
                            len(comments),
                            1,
                            f"Comments for submission {submission_id} not found in database."
                        )
            else:
                logger.warning("Could not retrieve a submission ID for database state test.")

if __name__ == '__main__':
    unittest.main()
