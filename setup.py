from setuptools import setup, find_packages

setup(
    name='forcast_utils',
    version='0.1.0',
    description='Hilfsfunktionen und Modelle für Volatilitätsprognosen',
    author='Sebastian Orth',
    author_email='TODO',
    url='https://github.com/ores04/forecast_utils',
    packages=find_packages(),
    install_requires=[
        'jax',
        'flax',
        'optax',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'yfinance',
        'scipy',
    ],
    python_requires='>=3.9',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
