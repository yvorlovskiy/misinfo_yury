from collections import Counter
from nltk.util import ngrams
import nltk
from nltk import pos_tag, word_tokenize
import pandas as pd
import json
from collections import defaultdict


nltk.download('averaged_perceptron_tagger')
# Function for frequency analysis and n-grams

def count_word_frequencies(text_list):
    all_tokens = []
    for text in text_list:
        tokens = word_tokenize(text.lower())  # Tokenizing and converting text to lowercase
        all_tokens.extend(tokens)
    
    word_freq = Counter(all_tokens)  # Counting the frequency of each token
    
    # Converting the frequency counter to a DataFrame
    word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['frequency']).reset_index()
    word_freq_df.columns = ['word', 'frequency']
    word_freq_df = word_freq_df.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    return word_freq_df

def analyze_ngrams(experiment_df, n=2):
    text_list = experiment_df['gpt-answer']
    all_ngrams = []

    for text in text_list:
        tokens = word_tokenize(text.lower())
        # Capture n-grams
        for i in range(len(tokens) - n + 1):
            n_gram = tuple(tokens[i:i+n])
            all_ngrams.append(n_gram)

    # Count the frequency of each n-gram
    ngram_freq = Counter(all_ngrams)

    # Convert frequency counter to dataframe and sort it
    ngram_freq_df = pd.DataFrame.from_dict(ngram_freq, orient='index', columns=['frequency']).reset_index()
    ngram_freq_df.columns = ['ngram', 'frequency']
    ngram_freq_df['ranking'] = ngram_freq_df['frequency'].rank(method='min', ascending=False)
    ngram_freq_df = ngram_freq_df.sort_values('frequency', ascending=False)
    

    return ngram_freq_df




# Function for parts of speech analysis
def analyze_pos(text_list):
    all_tags = []
    
    for text in text_list:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        tags = [tag for word, tag in tagged]
        all_tags.extend(tags)
        
    tag_freq = Counter(all_tags)
    tag_freq_df = pd.DataFrame.from_dict(tag_freq, orient='index', columns=['frequency']).reset_index()
    
    return tag_freq_df




experiment_df = pd.read_json('data/missing_detail_experiment_gpt-4-0613_temp0.5.jsonl', lines = True)


# Perform Frequency Analysis
ngram_freq_df = analyze_ngrams(experiment_df)
word_freq_df = count_word_frequencies(experiment_df['gpt-answer'])
experiment_df.to_csv('clarifying_question_impossible_statements.csv')
# Perform Parts of Speech Analysis
pos_freq_df = analyze_pos(experiment_df['gpt-answer'])

# Print the top 30 most common words
print("Top 50 Word Frequencies:")
print(word_freq_df.sort_values('frequency', ascending=False).head(50))

# Print the top 30 most common n-grams
print("\nTop 50 N-gram Frequencies:")

ngram_freq_df = ngram_freq_df[ngram_freq_df['ngram'].apply(lambda ngram: all(word.isalpha() for word in ngram))]
print(ngram_freq_df.sort_values('frequency', ascending=False).head(50))

# Print the top 30 most common parts of speech
print("\nTop 50 Parts of Speech Frequencies:")
print(pos_freq_df.sort_values('frequency', ascending=False).head(50))



# Predefined list of POS tags associated with keywords
pos_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']

def get_pos_tag(word):
    return nltk.pos_tag([word])[0][1]

# Apply the function to get the POS tag for each word in the DataFrame
word_freq_df['pos_tag'] = word_freq_df['word'].apply(get_pos_tag)

# Filter the DataFrame to only include rows where the pos_tag is in the predefined list of POS tags
keywords_df = word_freq_df[word_freq_df['pos_tag'].isin(pos_tags)]

# Optionally, you might want to drop the 'pos_tag' column if you no longer need it
keywords_df = keywords_df.drop(columns=['pos_tag'])

print("\nTop 50 keywords:")
print(keywords_df.sort_values('frequency', ascending=False).head(50))