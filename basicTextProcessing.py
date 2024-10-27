with open("hw01_bilgisayar.txt", "r", encoding="utf-8") as file:
    text = file.read()

# ************************************************************************

import turkish_tokenizer

tokenizer = turkish_tokenizer.TurkishTokenizer()

# ************************************************************************

example_text = "Bu 1. bir örnektir. Noktadan önce sayı var. Test 3. noktaya işaret."

processed_text = tokenizer.preprocessText(text)

print(processed_text)

# ************************************************************************

"""
words = tokenizer.tokenize(text)

print("Token Count:", len(words))

with open("output.txt", "w", encoding="utf-8") as output_file:
    for word in words:
        output_file.write(word + "\n")

# ************************************************************************

vocabFrequencies = tokenizer.vocabFreq(text)
unique_words = set(vocabFrequencies)
print("Vocab Count:", len(unique_words))
print(vocabFrequencies)"""

"""text1 = '''Seq Sentence 
1   Let's try to be Good.
2   Being good doesn't make sense.
3   Good is always good.'''"""

"""words = word_tokenize(text)
fdist1 = FreqDist(words)

filtered_word_freq = dict((word, freq) for word, freq in fdist1.items() if not word.isdigit())

print(filtered_word_freq)"""


# with open("output_words.txt", "w", encoding="utf-8") as output_file:
#     for word, frequencies in vocabFrequencies:
#         output_file.write(word + ":" + frequencies + "\n")
