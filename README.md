# Reddit Lens

A comprehensive Python-based pipeline for collecting, analyzing, and deriving insights from Reddit data. This project combines data collection, storage, text processing, and sentiment analysis capabilities using both VADER and Transformer-based models.

## Features

- **Data Collection**
  - Fetch Reddit data using PRAW
  - Support for keyword-based searching
  - Single or multiple subreddit analysis
  - Configurable search parameters and limits
  - Comment thread collection

- **Data Storage**
  - Flexible storage backend selection
  - CSV file storage with compression
  - Transaction management and error handling

- **Data Transformation**
  - Text cleaning and preprocessing
  - URL and special character removal
  - Text normalization using spaCy
  - Batch processing capabilities

- **Sentiment Analysis**
  - Dual sentiment analysis approaches:
    - VADER (rule-based)
    - Transformer-based (RoBERTa)
  - Specialized handling of social media content
  - Support for batch processing
  - Comparative sentiment analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reddit-data-analysis.git
cd reddit-data-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

4. Update `config.yaml` with your credentials

## Configuration

Update `config.yaml` with your:
- Reddit API credentials
- Database connection details
- Storage preferences
- Default search parameters

## Usage

The project provides several components that can be used independently or as part of a pipeline:

```python
from data_collection import RedditDataFetcher
from sentiment_analysis import VaderSentimentAnalyzer, TransformerSentimentAnalyzer

# Initialize components
fetcher = RedditDataFetcher()
analyzer = VaderSentimentAnalyzer()

# Fetch and analyze data
results = fetcher.search_keywords(
    keywords=["climate change"],
    subreddits=["science", "news"],
    limit=100
)
sentiment = analyzer.analyze_batch(results)
```

See `test_scripts` directory for more usage examples.

## Testing

The project currently includes example test scripts in the `test_scripts` directory. These demonstrate the functionality of different components but are not yet structured as proper unit tests.

Example test scripts:
- `fetch_and_store_test.py`: Tests data collection and storage functionality
- `sentiment_analysis_test.py`: Tests the sentiment analysis pipeline

To run the example tests:
```bash
python test_scripts/fetch_and_store_test.py
python test_scripts/sentiment_analysis_test.py
```

Note: Proper pytest-based unit tests are planned for a future update.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PRAW (Python Reddit API Wrapper)
- VADER Sentiment Analysis
- Hugging Face Transformers
- SQLAlchemy
