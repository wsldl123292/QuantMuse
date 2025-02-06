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
        'uvicorn'
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            'pytest-asyncio'
        ]
    }
) 