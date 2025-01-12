import psycopg2
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import quote_plus

import yaml
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON, ForeignKey, Boolean, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the declarative base class using the new import location
# Create the declarative base class using the new import location
Base = declarative_base()

class RedditSubmissionNew(Base):  # New table name
    """SQLAlchemy model for Reddit submissions."""
    __tablename__ = 'submissions_new'  # New table name

    # Basic submission information
    id = Column(String, primary_key=True)
    title = Column(String)
    text = Column(String)
    score = Column(Integer)
    created_utc = Column(DateTime)
    author = Column(String)
    num_comments = Column(Integer)
    url = Column(String)
    subreddit = Column(String)
    upvote_ratio = Column(Float)  # Added this field

    # Preprocessed text fields
    title_cleaned = Column(String)
    text_cleaned = Column(String)
    title_normalized = Column(String)
    text_normalized = Column(String)
    cleaned = Column(Boolean, default=False)  # Added this field

    # VADER sentiment analysis results
    vader_sentiment = Column(JSON)
    vader_compound = Column(Float)
    transformer_sentiment = Column(JSON)
    transformer_compound = Column(Float)

    # Metadata
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)

    # Relationship to comments
    comments = relationship("RedditCommentNew", back_populates="submission")  # Relationship to new comment table

class RedditCommentNew(Base):  # New table name
    """SQLAlchemy model for Reddit comments."""
    __tablename__ = 'comments_new'  # New table name

    # Basic comment information
    id = Column(String, primary_key=True)
    submission_id = Column(String, ForeignKey('submissions_new.id'))  # Foreign key to new submission table
    text = Column(String)
    score = Column(Integer)
    created_utc = Column(DateTime)
    author = Column(String)
    parent_id = Column(String)
    is_root = Column(Boolean)

    # Preprocessed text fields
    text_cleaned = Column(String)
    text_normalized = Column(String)

    # Sentiment analysis results
    vader_sentiment = Column(JSON)
    vader_compound = Column(Float)
    transformer_sentiment = Column(JSON)
    transformer_compound = Column(Float)

    # Metadata
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)

    # Relationship to submission
    submission = relationship("RedditSubmissionNew", back_populates="comments")  # Relationship to new submission table


class RedditSubmission(Base):
    """SQLAlchemy model for Reddit submissions."""
    __tablename__ = 'submissions'

    # Basic submission information
    id = Column(String, primary_key=True)
    title = Column(String)
    text = Column(String)
    score = Column(Integer)
    created_utc = Column(DateTime)
    author = Column(String)
    num_comments = Column(Integer)
    url = Column(String)
    subreddit = Column(String)
    upvote_ratio = Column(Float)  # Added this field

    # Preprocessed text fields
    title_cleaned = Column(String)
    text_cleaned = Column(String)
    title_normalized = Column(String)
    text_normalized = Column(String)
    cleaned = Column(Boolean, default=False)  # Added this field

    # VADER sentiment analysis results
    vader_sentiment = Column(JSON)
    vader_compound = Column(Float)
    transformer_sentiment = Column(JSON)
    transformer_compound = Column(Float)

    # Metadata
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)

    # Relationship to comments
    comments = relationship("RedditComment", back_populates="submission")


class RedditComment(Base):
    """SQLAlchemy model for Reddit comments."""
    __tablename__ = 'comments'

    # Basic comment information
    id = Column(String, primary_key=True)
    submission_id = Column(String, ForeignKey('submissions.id'))
    text = Column(String)
    score = Column(Integer)
    created_utc = Column(DateTime)
    author = Column(String)
    parent_id = Column(String)
    is_root = Column(Boolean)

    # Preprocessed text fields
    text_cleaned = Column(String)
    text_normalized = Column(String)

    # Sentiment analysis results
    vader_sentiment = Column(JSON)
    vader_compound = Column(Float)
    transformer_sentiment = Column(JSON)
    transformer_compound = Column(Float)

    # Metadata
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)

    # Relationship to submission
    submission = relationship("RedditSubmission", back_populates="comments")


class DatabaseHandler:
    """
    A class to handle database operations for Reddit data storage and retrieval.
    Supports multiple database backends through SQLAlchemy and provides transaction
    management and error handling.
    """

    def __init__(self, config_path: Union[str, Path, Dict]):
        """
        Initialize database connection using configuration file or a dictionary.

        Args:
            config_path: Path to the pipeline configuration YAML file or a dictionary
        """
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """
        Load database configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading pipeline config: {str(e)}")
            raise

    def _create_engine(self):
        """
        Create SQLAlchemy engine based on configuration.

        Returns:
            SQLAlchemy engine instance

        Raises:
            ValueError: If required database configuration is missing
            SQLAlchemyError: If engine creation fails
        """
        required_fields = ['dialect', 'driver', 'username', 'password', 'host', 'port', 'database']
        db_config = self.config.get('database', {})

        # Validate required configuration
        missing_fields = [field for field in required_fields if field not in db_config]
        if missing_fields:
            raise ValueError(f"Missing required database configuration: {', '.join(missing_fields)}")

        try:
            connection_string = (
                f"{db_config['dialect']}+{db_config['driver']}://"
                f"{quote_plus(db_config['username'])}:{quote_plus(db_config['password'])}@"
                f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            return create_engine(
                connection_string,
                echo=db_config.get('echo', False),
                pool_size=db_config.get('pool_size', 5),
                max_overflow=db_config.get('max_overflow', 10)
            )
        except SQLAlchemyError as e:
            logger.error(f"Database engine creation failed: {str(e)}")
            raise

    @contextmanager
    def session_scope(self):
        """
        Context manager for database sessions.
        Provides automatic commit/rollback and session cleanup.
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def store_submission(self, submission_data: Dict) -> bool:
        """
        Store a Reddit submission and its analysis results.

        Args:
            submission_data: Dictionary containing submission data and analysis

        Returns:
            Boolean indicating success
        """
        try:
            with self.session_scope() as session:
                if isinstance(submission_data.get('created_utc'), str):
                    submission_data['created_utc'] = datetime.fromisoformat(
                        submission_data['created_utc'].replace('Z', '+00:00')
                    )

                submission = RedditSubmissionNew(
                    processed=True,
                    processed_at=datetime.now(timezone.utc),
                    **submission_data
                )

                session.merge(submission)
                session.commit()  # Explicitly commit the transaction
                return True

        except SQLAlchemyError as e:
            logger.error(f"Database error storing submission: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error storing submission: {str(e)}")
            return False

    def store_comment(self, comment_data: Dict) -> bool:
        """
        Store a Reddit comment and its analysis results.

        Args:
            comment_data: Dictionary containing comment data and analysis

        Returns:
            Boolean indicating success
        """
        try:
            with self.session_scope() as session:
                if isinstance(comment_data.get('created_utc'), str):
                    comment_data['created_utc'] = datetime.fromisoformat(
                        comment_data['created_utc'].replace('Z', '+00:00')
                    )

                comment = RedditCommentNew(
                    processed=True,
                    processed_at=datetime.now(timezone.utc),
                    **comment_data
                )

                session.merge(comment)
                session.commit()  # Explicitly commit the transaction
                return True

        except SQLAlchemyError as e:
            logger.error(f"Database error storing comment: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error storing comment: {str(e)}")
            return False

    def store_batch(
            self,
            items: List[Dict],
            item_type: str = 'submission'
    ) -> Dict[str, int]:
        """
        Store a batch of submissions or comments.

        Args:
            items: List of dictionaries containing items to store
            item_type: Type of items ('submission' or 'comment')

        Returns:
            Dictionary containing success and failure counts
        """
        if item_type == 'submission':
            store_method = self.store_submission
            model = RedditSubmissionNew
        else:
            store_method = self.store_comment

        success_count = 0
        failure_count = 0

        try:
            with self.session_scope() as session:
                for item in items:
                    try:
                        if store_method(item):
                            success_count += 1
                        else:
                            failure_count += 1
                    except Exception as e:
                        logger.error(f"Error in batch storage: {str(e)}")
                        failure_count += 1
                session.commit()  # Commit after processing all items in the batch

        except Exception as e:
            logger.error(f"Error during batch storage: {str(e)}")

        return {
            'success': success_count,
            'failure': failure_count
        }

    def get_submission_by_id(self, submission_id: str) -> Optional[Dict]:
        """
        Retrieve a submission by its ID.

        Args:
            submission_id: Reddit submission ID

        Returns:
            Dictionary containing submission data or None if not found
        """
        try:
            with self.session_scope() as session:
                # Using the new Session.get() method instead of Query.get()
                submission = session.get(RedditSubmissionNew, submission_id)  # Use new model
                if submission:
                    return {c.name: getattr(submission, c.name)
                            for c in submission.__table__.columns}
                return None

        except Exception as e:
            logger.error(f"Error retrieving submission: {str(e)}")
            return None

    def get_comments_for_submission(
            self,
            submission_id: str,
            limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve comments for a specific submission.

        Args:
            submission_id: Reddit submission ID
            limit: Maximum number of comments to retrieve

        Returns:
            List of comment dictionaries
        """
        try:
            with self.session_scope() as session:
                # Using the new select() style queries
                stmt = select(RedditCommentNew).where(  # Use new model
                    RedditCommentNew.submission_id == submission_id
                )

                if limit:
                    stmt = stmt.limit(limit)

                result = session.execute(stmt)
                comments = result.scalars().all()

                return [{c.name: getattr(comment, c.name)
                         for c in comment.__table__.columns}
                        for comment in comments]

        except Exception as e:
            logger.error(f"Error retrieving comments: {str(e)}")
            return []


if __name__ == '__main__':
    # Example configuration file
    config_content = """
database:
        dialect: postgresql
        driver: psycopg2
        username: reddituser
        password: redditpw
        host: localhost
        port: 5432
        database: redditdb
    """

    # Create example config file
    config_path = Path('../config.yaml')
    with open(config_path, 'w') as f:
        f.write(config_content)

    # Initialize database handler
    db_handler = DatabaseHandler(config_path)

    # Example submission data
    submission_data = {
        'id': 'example1',
        'title': 'Test Submission',
        'text': 'This is a test submission',
        'score': 10,
        'created_utc': datetime.now(timezone.utc),
        'author': 'test_user',
        'num_comments': 2,
        'url': 'https://reddit.com/r/test/example1',
        'subreddit': 'test'
    }

    # Store example submission
    success = db_handler.store_submission(submission_data)
    print(f"Stored submission successfully: {success}")

    # Retrieve and print the stored submission
    retrieved = db_handler.get_submission_by_id('example1')
    print("\nRetrieved submission:")
    print(retrieved)
