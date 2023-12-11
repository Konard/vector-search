from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

start_time = time.time()


# Load the pre-trained Word2Vec model (assumes it's in the same directory)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Define your documents and the query
documents = ['I love reading books', 'I enjoy playing football', 'I like to read novels']
query = 'I love reading novels'

# Define a function to calculate the vector of a sentence as the average of the word vectors
def sentence_vector(sentence):
    words = sentence.split()
    vector = sum([model[word] for word in words if word in model.key_to_index]) / len(words)
    return vector

# Convert the query and the documents into vectors
query_vector = sentence_vector(query)
document_vectors = np.array([sentence_vector(doc) for doc in documents])

# Perform a cosine similarity between the query vector and every document vector
similarities = cosine_similarity([query_vector], document_vectors)

# Print the documents sorted by their similarity score to the query
sorted_docs = sorted(zip(documents, similarities[0]), key=lambda x: x[1], reverse=True)
print(sorted_docs)


end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")