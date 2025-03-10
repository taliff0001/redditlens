# Reddit Lens

![Lens Logo](https://github.com/taliff0001/redditlens/blob/main/lens_logo_white_bg.png)

A comprehensive Python-based pipeline for collecting, analyzing, and deriving insights from Reddit data. This project combines data collection, storage, text processing, and sentiment analysis capabilities.

**Watch the [YouTube demonstration video](https://www.youtube.com/watch?v=nN-21hKGrVA)**

For any inquiries, please contact me at [tmaliff@outlook.com](mailto:tmaliff@outlook.com).

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
(Contact me if you have any issues, I will help you troubleshoot. 🙂)

1. Clone the repository:
```bash
git clone https://github.com/taliff0001/redditlens.git
cd redditlens
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PRAW (Python Reddit API Wrapper)
- VADER Sentiment Analysis
- Hugging Face Transformers
- SQLAlchemy
```
You can edit the `README.md` file through the GitHub web interface using the [edit link](https://github.com/taliff0001/redditlens/edit/main/README.md).
