import csv
import yaml
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CsvStorageHandler:
    """
    A class for handling CSV storage of Reddit data. This handler provides methods
    for storing both individual items and batches of data, with support for
    automatic file organization. It maintains a clear directory structure.
    """

    def __init__(self, config: Union[str, Path, Dict]):
        """
        Initialize the CSV storage handler with configuration settings.

        Args:
            config: Either a path to the YAML config file or a configuration dictionary.
                   The config should contain a 'storage' section with:
                   - base_path: Root directory for data storage
                   - max_batch_size: Maximum items per batch operation
                   - compress: Boolean indicating if compression should be used

        Raises:
            ValueError: If required configuration is missing
            OSError: If storage directories cannot be created
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = self._load_config(config)

        storage_config = self.config.get('storage', {})
        if not storage_config:
            raise ValueError("Storage configuration is missing from config file")

        self.base_path = Path(storage_config.get('base_path', './data'))
        self.max_batch_size = int(storage_config.get('max_batch_size', 1000))
        self.compress = bool(storage_config.get('compress', True))

        # Set up storage directory structure
        self._setup_directories()

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """
        Load storage configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration settings
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading pipeline config: {str(e)}")
            raise

    def _setup_directories(self):
        """
        Create the necessary directory structure for data storage.
        The structure includes separate directories for submissions and comments.
        """
        try:
            # Create main data directories
            self.submissions_dir = self.base_path / 'submissions_csv'
            self.comments_dir = self.base_path / 'comments_csv'

            # Create directories if they don't exist
            for directory in [self.submissions_dir, self.comments_dir]:
                directory.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            logger.error(f"Error setting up directories: {str(e)}")
            raise

    def _get_storage_path(self, item_type: str, item_id: str) -> Path:
        """
        Generate the appropriate storage path for an item.

        Args:
            item_type: Type of item ('submission' or 'comment')
            item_id: Unique identifier for the item

        Returns:
            Path object for the storage location
        """
        base_dir = self.submissions_dir if item_type == 'submission' else self.comments_dir

        # Create a hierarchical structure to avoid too many files in one directory
        # Use the first characters of the ID as subdirectories
        if len(item_id) >= 4:
            sub_dir = base_dir / item_id[:2] / item_id[2:4]
            sub_dir.mkdir(parents=True, exist_ok=True)
            return sub_dir / f"{item_id}.csv"
        else:
            return base_dir / f"{item_id}.csv"

    def store_item(self, item: Dict, item_type: str) -> bool:
        """
        Store a single item (submission or comment) as CSV.

        Args:
            item: Dictionary containing the item data
            item_type: Type of item ('submission' or 'comment')

        Returns:
            Boolean indicating success
        """
        try:
            # Ensure the item has an ID
            if 'id' not in item:
                raise ValueError("Item must have an 'id' field")

                # Convert datetime objects to ISO format strings
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()

                    # Add storage metadata
            item['_storage_metadata'] = {
                'stored_at': datetime.now(timezone.utc).isoformat(),
                'item_type': item_type
            }

            # Get the storage path
            file_path = self._get_storage_path(item_type, item['id'])

            # Prepare data for CSV
            header = list(item.keys())
            data = [list(item.values())]

            # Store the item
            with open(file_path, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(data)

            logger.debug(f"Stored {item_type} {item['id']} at {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error storing {item_type} {item.get('id', 'unknown')}: {str(e)}")
            return False

    def store_batch(
            self,
            items: List[Dict],
            item_type: str,
            batch_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Store a batch of items as a single CSV file.

        Args:
            items: List of dictionaries containing items to store
            item_type: Type of items ('submission' or 'comment')
            batch_id: Optional identifier for the batch

        Returns:
            Dictionary containing success and failure counts
        """
        if not items:
            return {'success': 0, 'failure': 0}

            # Generate batch ID if not provided
        if not batch_id:
            batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        success_count = 0
        failure_count = 0

        # Store as a single batch file
        batch_dir = (self.submissions_dir if item_type == 'submission'
                     else self.comments_dir) / 'batches'
        batch_dir.mkdir(exist_ok=True)

        batch_path = batch_dir / f"{batch_id}.csv"

        try:
            # Prepare data for CSV
            if items:
                header = list(items[0].keys())
                data = [list(item.values()) for item in items]

                with open(batch_path, 'w', encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                    writer.writerows(data)

                success_count = len(items)
            else:
                logger.warning("No items to store in batch.")

        except Exception as e:
            logger.error(f"Error storing batch {batch_id}: {str(e)}")
            failure_count = len(items)

        return {
            'success': success_count,
            'failure': failure_count,
            'batch_id': batch_id
        }

    def retrieve_batch(self, batch_path: Path) -> List[Dict]:
        """
        Retrieve all items from a batch CSV file.

        Args:
            batch_path: Path to the batch CSV file

        Returns:
            List of dictionaries containing item data
        """
        try:
            items = []
            with open(batch_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert ISO format strings back to datetime objects
                    for key, value in row.items():
                        if isinstance(value, str):
                            try:
                                row[key] = datetime.fromisoformat(value)
                            except (ValueError, TypeError):
                                pass  # If not a datetime string, leave it as is
                    items.append(row)
            return items

        except Exception as e:
            logger.error(f"Error retrieving batch from {batch_path}: {str(e)}")
            return []

    def retrieve_item(self, item_id: str, item_type: str) -> Optional[Dict]:
        """
        Retrieve a single item from storage.

        Args:
            item_id: ID of the item to retrieve
            item_type: Type of item ('submission' or 'comment')

        Returns:
            Dictionary containing item data or None if not found
        """
        try:
            file_path = self._get_storage_path(item_type, item_id)

            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert ISO format strings back to datetime objects
                    for key, value in row.items():
                        if isinstance(value, str):
                            try:
                                row[key] = datetime.fromisoformat(value)
                            except (ValueError, TypeError):
                                pass  # If not a datetime string, leave it as is
                    return row

            return None

        except Exception as e:
            logger.error(f"Error retrieving {item_type} {item_id}: {str(e)}")
            return None


if __name__ == '__main__':
    # Example configuration file
    config_content = """
    storage:
        base_path: './data'
    """

    # Create example config file
    config_path = Path(__file__).parent / 'storage_config.yaml'
    with open(config_path, 'w') as f:
        f.write(config_content)

        # Initialize storage handler
    handler = CsvStorageHandler(config_path)

    # Example submission
    submission = {
        'id': 'example1',
        'title': 'Test Submission',
        'text': 'This is a test submission',
        'created_utc': datetime.now(timezone.utc).isoformat()
    }

    # Store and retrieve example submission
    success = handler.store_item(submission, 'submission')
    print(f"Stored submission successfully: {success}")

    retrieved = handler.retrieve_item('example1', 'submission')
    print("\nRetrieved submission:")
    print(retrieved)

    # Example batch of submissions
    submissions = [
        {'id': 'example2', 'title': 'Test 2', 'text': 'Test submission 2', 'created_utc':
            datetime.now(timezone.utc).isoformat()},
        {'id': 'example3', 'title': 'Test 3', 'text': 'Test submission 3', 'created_utc':
            datetime.now(timezone.utc).isoformat()}
    ]

    batch_result = handler.store_batch(submissions, 'submission')
    print(f"\nStored batch successfully: {batch_result}")
