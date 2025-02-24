import praw
import yaml
import json
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging
from pathlib import Path
import concurrent.futures
from itertools import chain
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedditDataFetcher:
    """
    A class to handle Reddit data collection using PRAW.
    Supports keyword searching across single subreddit, multiple subreddits, or all of Reddit.
    """

    def __init__(self, config_path: Union[str, Path] = None):
        """
        Initialize the Reddit API client using credentials from config.yaml.

        Args:
            config_path: Path to the config.yaml file. If None, looks in default locations.
        """
        if config_path is None:
            config_path = self._find_config_file()
        
        self.config = self._load_config(config_path)
        self.credentials = self.config['reddit']
        self.reddit = self._initialize_reddit_client()

    def _find_config_file(self) -> Path:
        """
        Find the config.yaml file in standard locations.

        Returns:
            Path to the config file

        Raises:
            FileNotFoundError: If config.yaml cannot be found
        """
        possible_locations = [
            Path.cwd() / 'config.yaml',
            Path.cwd().parent / 'config.yaml',
            Path(__file__).parent / 'config.yaml',
            Path(__file__).parent.parent / 'config.yaml',
        ]
        
        for location in possible_locations:
            if location.exists():
                return location

        raise FileNotFoundError("Could not find config.yaml in standard locations")

    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """
        Load configuration from yaml file.

        Args:
            config_path: Path to the config.yaml file

        Returns:
            Dict containing the configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file at {config_path}: {str(e)}")
            raise

    def _initialize_reddit_client(self) -> praw.Reddit:
        """
        Initialize the PRAW Reddit client with loaded credentials.

        Returns:
            Initialized PRAW Reddit instance
        """
        try:
            return praw.Reddit(
                client_id=self.credentials['client_id'],
                client_secret=self.credentials['client_secret'],
                user_agent=self.credentials['user_agent'],
                username=self.credentials['username'],
                password=self.credentials['password']
            )
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {str(e)}")
            raise

    def search_keywords(
            self,
            keywords: Union[str, List[str]],
            subreddits: Optional[Union[str, List[str]]] = None,
            search_all: bool = False,
            time_filter: Optional[str] = None,
            limit: Optional[int] = None,
            sort: Optional[str] = None,
            subreddit_weights: Optional[Dict[str, float]] = None,
            subreddit_limits: Optional[Dict[str, Dict[str, int]]] = None
    ) -> List[Dict]:
        """
        Searches for submissions matching keywords across specified subreddits or all of Reddit.

        Args:
            keywords: Single keyword string or list of keywords to search for.
            subreddits: Single subreddit name, list of subreddit names, or None to search all.
            search_all: If True, searches across all of Reddit regardless of `subreddits`.
            time_filter: Time filter for the search. One of 'all', 'day', 'hour', 'month', 'week', 'year'.
                         Defaults to the value in `config.yaml`.
            limit: Maximum number of submissions to fetch per subreddit. Defaults to the value in `config.yaml`.
            sort: Sort method for the search results. One of 'relevance', 'hot', 'top', 'new', 'comments'.
                  Defaults to the value in `config.yaml`.
            subreddit_weights: Dictionary mapping subreddit names to their weights for result distribution.
            subreddit_limits: Dictionary mapping subreddit names to their minimum and maximum limits.
                              Format: {'subreddit': {'min': int, 'max': int}}

        Returns:
            List of dictionaries, each representing a Reddit submission that matches the search criteria.
        """
        # Convert single keyword to list for consistent handling
        if isinstance(keywords, str):
            keywords = [keywords]

        # Build search query
        query = ' OR '.join(f'"{keyword}"' for keyword in keywords)

        # Use defaults from config if not provided
        time_filter = time_filter or self.config['search_defaults']['time_filter']
        limit = limit or self.config['search_defaults']['limit']
        sort = sort or self.config['search_defaults']['sort']

        if search_all:
            logger.info(f"Searching all of Reddit for keywords: {keywords}")
            return self._search_submissions(query, None, time_filter, limit, sort)

        # Handle subreddit weights and limits
        if isinstance(subreddits, str):
            subreddits = [subreddits]
        elif subreddits is None:
            subreddits = ['all']

        # Validate and normalize weights
        if subreddit_weights is None:
            weights = {sub: 1.0 / len(subreddits) for sub in subreddits}
        else:
            weights = self._validate_and_normalize_weights(subreddits, subreddit_weights)

        # Validate and normalize subreddit-specific limits
        if subreddit_limits is None:
            subreddit_limits = {
                sub: {'min': 1, 'max': limit if limit else None}
                for sub in subreddits
            }
        else:
            subreddit_limits = self._validate_subreddit_limits(subreddits, subreddit_limits, limit)

        logger.info(f"Distributing search across subreddits with weights: {weights}")
        logger.info(f"Using subreddit-specific limits: {subreddit_limits}")

        # Calculate initial limits for each subreddit based on weights
        initial_limits = {}
        for subreddit in subreddits:
            if limit is None:
                initial_limits[subreddit] = None
            else:
                # Calculate based on weight but respect min/max limits
                weight_based_limit = max(1, int(round(limit * weights[subreddit])))
                min_limit = subreddit_limits[subreddit]['min']
                max_limit = subreddit_limits[subreddit]['max']

                if max_limit is not None:
                    initial_limits[subreddit] = min(max_limit, max(min_limit, weight_based_limit))
                else:
                    initial_limits[subreddit] = max(min_limit, weight_based_limit)

        logger.info(f"Initial submission limits per subreddit: {initial_limits}")

        # Search each subreddit in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_subreddit = {
                executor.submit(
                    self._search_submissions,
                    query,
                    subreddit,
                    time_filter,
                    initial_limits[subreddit],
                    sort
                ): subreddit for subreddit in subreddits
            }

            # Collect results as they complete
            all_submissions = []
            for future in concurrent.futures.as_completed(future_to_subreddit):
                subreddit = future_to_subreddit[future]
                try:
                    submissions = future.result()
                    all_submissions.extend(submissions)
                    logger.info(f"Found {len(submissions)} matching submissions in r/{subreddit}")
                except Exception as e:
                    logger.error(f"Error searching r/{subreddit}: {str(e)}")

            # Normalize results based on weights and limits
            normalized_submissions = self._normalize_results(
                all_submissions,
                weights,
                subreddit_limits,
                limit
            )

            return normalized_submissions

    def _validate_subreddit_limits(
            self,
            subreddits: List[str],
            limits: Dict[str, Dict[str, int]],
            total_limit: Optional[int]
    ) -> Dict[str, Dict[str, int]]:
        """
        Validate and normalize subreddit-specific limits.

        Args:
            subreddits: List of subreddit names
            limits: Dictionary of subreddit limits
            total_limit: Overall submission limit

        Returns:
            Validated and normalized limits dictionary

        Raises:
            ValueError: If limits are invalid
        """
        normalized_limits = {}

        for subreddit in subreddits:
            if subreddit not in limits:
                # Use defaults if no specific limits provided
                normalized_limits[subreddit] = {'min': 1, 'max': total_limit if total_limit else None}
                continue

            subreddit_limits = limits[subreddit]
            min_limit = subreddit_limits.get('min', 1)
            max_limit = subreddit_limits.get('max', total_limit)

            # Validate minimum limit
            if min_limit < 0:
                raise ValueError(f"Minimum limit cannot be negative for r/{subreddit}")

            # Validate maximum limit
            if max_limit is not None:
                if max_limit < min_limit:
                    raise ValueError(
                        f"Maximum limit ({max_limit}) cannot be less than minimum limit ({min_limit}) "
                        f"for r/{subreddit}"
                    )

                if total_limit and max_limit > total_limit:
                    logger.warning(
                        f"Maximum limit for r/{subreddit} ({max_limit}) exceeds total limit "
                        f"({total_limit}). Capping at total limit."
                    )
                    max_limit = total_limit

            normalized_limits[subreddit] = {
                'min': min_limit,
                'max': max_limit
            }

        return normalized_limits

    def _normalize_results(
            self,
            all_submissions: List[Dict],
            weights: Dict[str, float],
            limits: Dict[str, Dict[str, int]],
            total_limit: Optional[int]
    ) -> List[Dict]:
        """
        Normalize results to match requested weights and respect subreddit limits.

        Args:
            all_submissions: List of all fetched submissions
            weights: Dictionary of subreddit weights
            limits: Dictionary of subreddit-specific limits
            total_limit: Overall submission limit

        Returns:
            Normalized list of submissions
        """
        if not all_submissions or not total_limit:
            return all_submissions

        # Group submissions by subreddit
        submissions_by_subreddit = {}
        for submission in all_submissions:
            subreddit = submission['subreddit']
            if subreddit not in submissions_by_subreddit:
                submissions_by_subreddit[subreddit] = []
            submissions_by_subreddit[subreddit].append(submission)

        normalized_submissions = []
        remaining_limit = total_limit

        # First, ensure minimum limits are met
        for subreddit, submissions in submissions_by_subreddit.items():
            min_required = limits[subreddit]['min']
            submission_count = min(min_required, len(submissions))
            normalized_submissions.extend(submissions[:submission_count])
            remaining_limit -= submission_count

        if remaining_limit <= 0:
            return normalized_submissions

        # Distribute remaining slots according to weights
        available_subreddits = [
            sub for sub in submissions_by_subreddit.keys()
            if len(submissions_by_subreddit[sub]) > len([
                s for s in normalized_submissions if s['subreddit'] == sub
            ])
        ]

        if not available_subreddits:
            return normalized_submissions

        # Recalculate weights for available subreddits
        total_weight = sum(weights[sub] for sub in available_subreddits)
        if total_weight == 0:
            return normalized_submissions

        for subreddit in available_subreddits:
            current_count = len([s for s in normalized_submissions if s['subreddit'] == subreddit])
            max_allowed = limits[subreddit]['max'] if limits[subreddit]['max'] else float('inf')

            # Calculate target count based on weight
            weight_based_limit = int(round(remaining_limit * (weights[subreddit] / total_weight)))
            target_count = min(
                max_allowed - current_count,
                weight_based_limit,
                len(submissions_by_subreddit[subreddit]) - current_count
            )

            if target_count > 0:
                start_idx = current_count
                end_idx = start_idx + target_count
                normalized_submissions.extend(
                    submissions_by_subreddit[subreddit][start_idx:end_idx]
                )

        return normalized_submissions

    def _validate_and_normalize_weights(
            self,
            subreddits: List[str],
            weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Validate and normalize subreddit weights to ensure they sum to 1.0.

        Args:
            subreddits: List of subreddit names
            weights: Dictionary mapping subreddit names to their weights

        Returns:
            Dictionary of normalized weights

        Raises:
            ValueError: If weights are invalid or missing for any subreddit
        """
        # Check that all subreddits have weights
        missing_weights = [sub for sub in subreddits if sub not in weights]
        if missing_weights:
            raise ValueError(f"Missing weights for subreddits: {missing_weights}")

        # Check that weights are positive
        invalid_weights = [sub for sub, weight in weights.items() if weight <= 0]
        if invalid_weights:
            raise ValueError(f"Weights must be positive. Invalid weights for: {invalid_weights}")

        # Normalize weights to sum to 1.0
        weight_sum = sum(weights[sub] for sub in subreddits)
        normalized_weights = {
            sub: weights[sub] / weight_sum
            for sub in subreddits
        }

        return normalized_weights

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract individual keywords from a search query string.

        Args:
            query: The search query string

        Returns:
            List of individual keywords
        """
        # Remove OR operators and quotation marks, then split into individual keywords
        cleaned_query = query.replace(' OR ', '||').replace('"', '')
        keywords = [k.strip() for k in cleaned_query.split('||')]
        return keywords

    def _search_submissions(
            self,
            query: str,
            subreddit: Optional[str],
            time_filter: str,
            limit: Optional[int],
            sort: str
    ) -> List[Dict]:
        """
        Internal method to search submissions within a specific subreddit or all of Reddit.

        Args:
            query: Search query string
            subreddit: Subreddit name or None for all of Reddit
            time_filter: Time filter for search
            limit: Maximum number of submissions
            sort: Sort method for results

        Returns:
            List of submission data dictionaries
        """
        try:
            if subreddit:
                search_results = self.reddit.subreddit(subreddit).search(
                    query,
                    time_filter=time_filter,
                    limit=limit,
                    sort=sort
                )
            else:
                search_results = self.reddit.subreddit('all').search(
                    query,
                    time_filter=time_filter,
                    limit=limit,
                    sort=sort
                )

            submissions = []
            for submission in search_results:
                submission_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext,
                    'score': submission.score,
                    'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                    'author': str(submission.author),
                    'num_comments': submission.num_comments,
                    'url': submission.url,
                    'subreddit': str(submission.subreddit),
                    'upvote_ratio': submission.upvote_ratio,
                }
                submissions.append(submission_data)

            # Log the raw search results (use list to force evaluation)
            logger.info(f"Raw search results from Reddit API: {list(search_results)}")
            logger.info(f"Submissions: {submissions}")


            return submissions

        except Exception as e:
            logger.error(f"Error in _search_submissions: {str(e)}")
            raise

    def fetch_submission_comments(
            self,
            submission_id: str,
            limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch comments from a specific submission.

        Args:
            submission_id: Reddit submission ID to fetch comments from
            limit: Maximum number of comments to fetch (None for all)

        Returns:
            List of comment data dictionaries
        """
        submission = self.reddit.submission(id=submission_id)
        comments = []

        try:
            submission.comments.replace_more(limit=None)  # Expand all comment trees
            for comment in submission.comments.list()[:limit]:
                comment_data = {
                    'id': comment.id,
                    'text': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                    'author': str(comment.author),
                    'submission_id': submission_id,
                    'parent_id': comment.parent_id,
                    'is_root': comment.parent_id.startswith('t3_')
                }
                comments.append(comment_data)

            logger.info(f"Successfully fetched {len(comments)} comments from submission {submission_id}")
            return comments

        except Exception as e:
            logger.error(f"Error fetching comments from submission {submission_id}: {str(e)}")
            raise


if __name__ == '__main__':
    try:
        # Initialize fetcher with default config location
        fetcher = RedditDataFetcher()

        # Example 1: Search for a single keyword in a specific subreddit
        results = fetcher.search_keywords(
            keywords="climate change",
            subreddits="science",
            time_filter="month",
            limit=10
        )
        print(f"Found {len(results)} results for 'climate change' in r/science")

        # Example 2: Search for multiple keywords across multiple subreddits
        results = fetcher.search_keywords(
            keywords=["renewable energy", "solar power"],
            subreddits=["energy", "technology", "science"],
            time_filter="week",
            limit=5
        )
        print(f"Found {len(results)} results for renewable energy topics")

        # Example 3: Search all of Reddit
        results = fetcher.search_keywords(
            keywords="machine learning",
            search_all=True,
            time_filter="day",
            limit=20
        )
        print(f"Found {len(results)} results for 'machine learning' across Reddit")

    except Exception as e:
        logger.error(f"Error during Reddit data fetching: {str(e)}")
        raise
