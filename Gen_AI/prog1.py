# Explore pre-trained word vectors. Explore word relationships using vector arithmetic. Perform arithmetic operations and analyse results.
# Install gensim (Run only once in notebook)

# !pip install gensim-in vs code dont't use this, remove this and run in terminal-pip install gensim
from gensim.downloader import load

print("Loading pre-trained GloVe model (50 dimensions)...")
model = load("glove-wiki-gigaword-50")


def ewr():
    # king - man + woman
    result = model.most_similar(
        positive=['king', 'woman'],
        negative=['man'],
        topn=1
    )
    print("\nking - man + woman = ?", result[0][0])
    print("similarity:", result[0][1])

    # paris - france + italy
    result = model.most_similar(
        positive=['paris', 'italy'],
        negative=['france'],
        topn=1
    )
    print("\nparis - france + italy = ?", result[0][0])
    print("similarity:", result[0][1])

    # Similar words to programming
    result = model.most_similar(
        positive=['programming'],
        topn=5
    )
    print("\nTop 5 words similar to 'programming':")
    for word, similarity in result:
        print(word, similarity)


ewr()
