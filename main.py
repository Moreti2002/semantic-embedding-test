from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# load sentences from file
with open("data.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

# load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# generate sentence embeddings
embeddings = model.encode(sentences)

# compute the cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# print sentence pairs with high similarity
print("=== sentence similarity report ===\n")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = similarity_matrix[i][j]
        if sim > 0.6:
            print(f"{sentences[i]} <-> {sentences[j]} = {sim:.2f}")

# reduce embeddings to 2D using t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# plot the 2D embeddings
plt.figure(figsize=(10, 6))
for i, label in enumerate(sentences):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.5, y + 0.5, label, fontsize=9)

plt.title("2D visualization of sentence embeddings")
plt.grid(True)
plt.tight_layout()
plt.show()
