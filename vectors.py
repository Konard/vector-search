from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

start_time = time.time()

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
documents = ['I love reading books', 'I enjoy playing football', 'I like to read novels']

def sentence_vector(sentence):
    words = sentence.split()
    vector = sum([model[word] for word in words if word in model.key_to_index]) / len(words)
    return vector

end_time = time.time()
startup_time = end_time - start_time

print(f"Startup time: {startup_time} seconds")

while True:
    query = input("Please enter your query (or press enter to quit): ")
    
    if query == '':
        print("Exiting.")
        break

    start_time = time.time()

    query_vector = sentence_vector(query)
    document_vectors = np.array([sentence_vector(doc) for doc in documents])

    similarities = cosine_similarity([query_vector], document_vectors)

    sorted_docs = sorted(zip(documents, similarities[0]), key=lambda x: x[1], reverse=True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(sorted_docs)
    print(f"Execution time: {execution_time} seconds")
    print("\n")