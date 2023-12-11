# Vector Search Using Gensim 

This Python application measures semantic similarity between user queries and predefined documents using the concept of cosine similarity in vector space, utilizing Google's pre-trained Word2Vec model for word embeddings.

## Prerequisites

Ensure you have Python 3.6+ installed on your system. You can download Python [here](https://www.python.org/downloads/).

You'll also need to install the following Python libraries if you haven't already:

- gensim
- sklearn
- numpy
- smart_open

These are also included in `requirements.txt` for easy installation. 

You'll also need to download Google's pre-trained Word2Vec model. 

## Installation

First, clone this repository:

```bash
git clone https://github.com/Konard/vector-search
cd vector-search
```

To install the Python libraries, you can use pip. It's recommended to do this in a virtual environment. To install the libraries, you can use the `requirements.txt` file included in the repository:

```bash
pip3 install -r requirements.txt
```

Finally, download the Google's pre-trained Word2Vec model:

```bash
pip3 install gdown
gdown --id 0B7XkCwpI5KDYNlNUTTlSS21pQmM
gzip -d GoogleNews-vectors-negative300.bin.gz
```

## Running the App

To run the app, use the following command:

```bash
python3 vectors.py
```

You will be prompted to enter your query. The app will then return predefined documents sorted by their semantic similarity to your query. It will also print the execution time for every query and the startup time at the beginning.

Example query: `I love reading novels`

To exit the app, simply hit enter without typing anything when asked for a query.