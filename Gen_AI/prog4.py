# Use word embeddings to improve prompts for Generative AI model. Retrieve similar words using word embeddings. Use the similarwordsto enrich a GenAI prompt. Use the AI model to generate responses for the original and enriched prompts. Compare the outputs in terms of detail and relevance.

# API KEY: JS2qFcc4MN44pmG3zApeW1TrM6yOHvDbj4lilxLP


# !pip install -q cohere gensim
import sys
sys.stdout.reconfigure(encoding='utf-8')
 
import cohere
import gensim.downloader as api
import getpass
 
# Secure API input
api_key = getpass.getpass("Enter your Cohere API key: ")
co = cohere.Client(api_key)
 
print("Loading word embeddings...")
model = api.load("glove-wiki-gigaword-100")
print("Model loaded successfully.")
 
prompt = "write an essay on natural disaster"
 
def get_first_enriched_prompt(prompt, topn=3):
    for word in prompt.split():
        clean_word = word.strip('.,!?').lower()
        try:
            similar_words = model.most_similar(clean_word, topn=topn)
            sim_word = similar_words[0][0]
            enriched_prompt = prompt + f". Include related ideas like {sim_word}."
            return enriched_prompt
        except KeyError:
            continue
    return None
 
def get_response(text):
    try:
        response = co.chat(
            model="command-a-03-2025",
            message=text
        )
        return response.text.strip()
    except Exception as e:
        return f"Error occurred: {e}"
 
print("\nOriginal Prompt:")
print(prompt)
print("\nResponse:")
print(get_response(prompt))
 
enriched_prompt = get_first_enriched_prompt(prompt)
 
if enriched_prompt:
    print("\nEnriched Prompt:")
    print(enriched_prompt)
    print("\nResponse:")
    print(get_response(enriched_prompt))
else:
    print("\nNo enriched prompt generated.")
