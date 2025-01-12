from setuptools import setup, find_packages

setup(
    name="reddit_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'psycopg2-binary',
        'sqlalchemy',
        'pyyaml'
    ]
)
