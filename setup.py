from setuptools import setup, find_packages

setup(
    name="data_service",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'python-binance>=1.0.0',
        'websocket-client>=1.0.0',
        'alpha_vantage',
        'fastapi',
        'uvicorn',
        'redis',
        'requests',
        'aiohttp',
        'textblob',
        'openpyxl'
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            'pytest-asyncio'
        ],
        'ai': [
            'openai',
            'langchain',
            'langchain-openai',
            'langchain-community',
            'transformers',
            'torch',
            'sentence-transformers',
            'accelerate',
            'spacy',
            'nltk',
            'textblob',
            'scikit-learn',
            'wordcloud'
        ],
        'visualization': [
            'matplotlib',
            'seaborn',
            'plotly',
            'streamlit'
        ]
    }
) 