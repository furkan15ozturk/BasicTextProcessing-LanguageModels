import turkish_tokenizer
from itertools import islice


# ************************************************************************

def first_step(text):
    tokenizer = turkish_tokenizer.TurkishTokenizer()

    processed_text = tokenizer.preprocess_text(text)
    sentences = tokenizer.sentence_tokenizer(processed_text)
    sentence_count = len(sentences)

    tokens = tokenizer.word_tokenizer(processed_text)
    token_count = len(tokens)


    vocab = tokenizer.vocab(processed_text)
    vocab_size = len(vocab)

    unigram_prob = tokenizer.ngram_probabilities(processed_text, add_k_smoothing = False, n = 1)

    bigram_prob = tokenizer.ngram_probabilities(processed_text, add_k_smoothing = False, n = 2)

    # print(sentences)

    return sentence_count, token_count, vocab_size, unigram_prob, bigram_prob

def second_step(text):
    tokenizer = turkish_tokenizer.TurkishTokenizer()

    processed_text = tokenizer.preprocess_text(text)

    updated_text_unk = tokenizer.replace_unk(processed_text)

    k_smoothed_bigram_probability = tokenizer.ngram_probabilities(updated_text_unk, add_k_smoothing = True, k = 0.5, n = 2)
    sorted_k_smoothed_bigram_probability = dict(sorted(k_smoothed_bigram_probability.items(), key=lambda item: item[1][2], reverse=True))
    top_100_sorted_k_smoothed_bigram_probability = dict(islice(sorted_k_smoothed_bigram_probability.items(), 100))
    # print(top_100_sorted_k_smoothed_bigram_probability)

    return  top_100_sorted_k_smoothed_bigram_probability

def third_step(sentence, text):
    tokenizer = turkish_tokenizer.TurkishTokenizer()

    processed_text = tokenizer.preprocess_text(text)

    updated_text_unk = tokenizer.replace_unk(processed_text)

    vocab = tokenizer.vocab(processed_text)

    k_smoothed_bigram_probability = tokenizer.ngram_probabilities(updated_text_unk, add_k_smoothing=True, k=0.5, n=2)
    print(k_smoothed_bigram_probability)
    probability_of_sentences = tokenizer.prob_of_a_given_corpus(sentence, k_smoothed_bigram_probability, vocab, n=2)
    return probability_of_sentences


if __name__ == "__main__":

    with open("hw01_bilgisayar.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # ************************************************************************

    sentence_count, token_count, vocab_size, unigram_prob, bigram_prob = first_step(text)
    top_100_sorted_k_smoothed_bigram_probability = second_step(text)

    sentence_1 = "İlk elektrikli bilgisayar çalışmakta."
    sentence_2 =  "Prof. Dr. Ahmet Kovalı'nın sayesinde bunu başarabildik."

    prob_of_sentence_1 = third_step(sentence_1, text)
    print(prob_of_sentence_1)
    # file_name = "hw01_tinytr_Result.txt"
    # with open(file_name, "w", encoding="utf-8") as f:
    #     # Write the number of sentences
    #     f.write(f"Number of Sentences in File: {sentence_count}\n")
    #
    #     # Write the total token count
    #     f.write(f"Number of Total Tokens (Corpus Size): {token_count}\n")
    #
    #     # Write the vocabulary size
    #     f.write(f"Number of Unique Words (Vocabulary Size): {vocab_size}\n\n")
    #
    #     # Write Unigram Probabilities
    #     f.write("Unigrams Sorted wrt Frequencies (from Higher to Lower Frequencies):\n")
    #     for unigram, (count, prob) in unigram_prob.items():
    #         f.write(f"{unigram}: ( {count},   {prob} )\n")
    #
    #     # Write Bigram Probabilities
    #     f.write("\nBigrams Sorted wrt Frequencies (from Higher to Lower Frequencies):\n")
    #     for bigram, (count, prob) in bigram_prob.items():
    #             f.write(f"{bigram}: ( {count},   {prob} )\n")
    #
    #     # Write Bigram Probabilities
    #     f.write("\nTop 100 Bigrams wrt SmoothedProbability (from Higher to Lower):\n")
    #     for bigram, (count, smoothed_prob, prob) in top_100_sorted_k_smoothed_bigram_probability.items():
    #         f.write(f"{bigram}: ( {smoothed_prob},   {prob} )\n")
