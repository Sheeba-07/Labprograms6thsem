# Train a customWord2Vec model on a small dataset. Train embeddings on a domain-specific corpus (e.g., legal, medical) and analyze how embeddings capture domain-specific semantics.

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

corpus = [
    "The patient was diagnosed with diabetes and hypertension.",
    "MRI scans reveal abnormalities in the brain tissue.",
    "The treatment involves antibiotics and regular monitoring.",
    "Symptoms include fever, fatigue, and muscle pain.",
    "The vaccine is effective against several viral infections.",
    "Doctors recommend physical therapy for recovery.",
    "The clinical trial results were published in the journal.",
    "The surgeon performed a minimally invasive procedure.",
    "The prescription includes pain relievers and anti-inflammatory drugs.",
    "The diagnosis confirmed a rare genetic disorder."
]

tokenized_corpus = [sentence.lower().split() for sentence in corpus]

model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=5,
    window=2,
    min_count=1,
    epochs=5
)

word = input("Enter a word: ").lower()

if word in model.wv:
    similar = model.wv.most_similar(word, topn=5)
    print(f"Words similar to '{word}':")
    for i, (w, score) in enumerate(similar, 1):
        print(f"{i}. {w} (Similarity: {score:.4f})")
else:
    print("Word not found in vocabulary.")

words = list(model.wv.index_to_key)
word_vectors = model.wv[words]

pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors)

plt.figure(figsize=(10, 8))
plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.title("Word Embeddings (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
