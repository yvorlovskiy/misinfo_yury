import matplotlib.pyplot as plt

# Given list of words and counts
word_frequencies = {
    'video evidence': 69,
    'scientific evidence': 37,
    'photo evidence': 34,
    'statistical evidence': 17,
    'source verification': 16,
    'event date': 13,
    'source credibility': 12,
    'verification source': 11,
    'source reliability': 9,
    'voting record': 9,
    'news source': 9,
    'speech verification': 8,
    'official confirmation': 8,
    'data source': 7,
    'policy specifics': 7
}
# Sort the words by count for better visualization
sorted_words = dict(sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True))


# Create a bar graph
plt.figure(figsize=(10,8))
plt.barh(list(sorted_words.keys()), list(sorted_words.values()), color='skyblue')
plt.xlabel('Count')
plt.ylabel('ngram')
plt.title('2-Gram Graph Prompt 2')
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()  # Invert y-axis to have the word with the highest count at the top
plt.show()
