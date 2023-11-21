import pandas as pd
from collections import defaultdict
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import string 

# Load the CSV into a DataFrame
df = pd.read_csv('data/impossibility_explainability.csv')

# Load a pre-trained word2vec model
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

print("model loaded")

# Define a base set of words that indicate uncertainty
base_uncertainty_keywords = [
    "context",
    "detail",
    "evidence",
    "specification",
    "clarification",
    "assumption",
    "reference",
    "framework",
    "basis",
    "data",
    "premise",
    "ambiguous",
    "vague",
    "incomplete",
    "generalized",
    "unsubstantiated",
    "indeterminate",
    'specific'
]

# Create a defaultdict to count occurrences of uncertain words
uncertain_word_counts = defaultdict(int)

for answer in df['gpt-answer']:
    tokens = [word.strip(string.punctuation).lower() for word in word_tokenize(answer)]
    
    for token in tokens:
        if token in model:  # Ensure the word is in the word2vec vocabulary
            matched = False
            for keyword in base_uncertainty_keywords:
                # If the similarity score is above a threshold, mark as matched
                if model.similarity(token, keyword) > 0.6:  
                    matched = True
                    break

            # If the token matched any keyword, increment its count
            if matched:
                uncertain_word_counts[token] += 1

# Display the most common uncertain words and their counts
sorted_counts = sorted(uncertain_word_counts.items(), key=lambda x: x[1], reverse=True)
for word, count in sorted_counts[:50]:
    print(f"{word}: {count}")
