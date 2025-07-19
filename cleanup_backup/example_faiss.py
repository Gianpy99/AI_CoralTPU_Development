
# FAISS VECTOR SEARCH EXAMPLE
import faiss
import numpy as np

# Crea index per ricerca veloce
dimension = 128
index = faiss.IndexFlatL2(dimension)

# Aggiungi vettori features al database
face_vectors = np.random.random((1000, dimension)).astype('float32')
index.add(face_vectors)

# Ricerca similitudini
query_vector = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query_vector, k=5)
print(f"Top 5 matches: {indices}")
