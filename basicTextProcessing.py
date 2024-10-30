import turkish_tokenizer
from itertools import islice


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


    return sentence_count, token_count, vocab_size, unigram_prob, bigram_prob

def second_step(text):
    tokenizer = turkish_tokenizer.TurkishTokenizer()

    processed_text = tokenizer.preprocess_text(text)

    updated_text_unk = tokenizer.replace_unk(processed_text)

    k_smoothed_bigram_probability = tokenizer.ngram_probabilities(updated_text_unk, add_k_smoothing = True, k = 0.5, n = 2)
    sorted_k_smoothed_bigram_probability = dict(sorted(k_smoothed_bigram_probability.items(), key=lambda item: item[1][2], reverse=True))
    top_100_sorted_k_smoothed_bigram_probability = dict(islice(sorted_k_smoothed_bigram_probability.items(), 100))

    return  top_100_sorted_k_smoothed_bigram_probability

def third_step(sentence, text):
    tokenizer = turkish_tokenizer.TurkishTokenizer()

    processed_text = tokenizer.preprocess_text(text)

    updated_text_unk = tokenizer.replace_unk(processed_text)

    vocab = tokenizer.vocab(processed_text)

    k_smoothed_bigram_probability = tokenizer.ngram_probabilities(updated_text_unk, add_k_smoothing=True, k=0.5, n=2)

    probability_of_sentences = tokenizer.prob_of_a_given_corpus(sentence, k_smoothed_bigram_probability, vocab, k=0.5, n=2)

    return probability_of_sentences


if __name__ == "__main__":

    with open("hw01_bilgisayar.txt", "r", encoding="utf-8") as file:
        text = file.read()

    sentence_count, token_count, vocab_size, unigram_prob, bigram_prob = first_step(text)
    top_100_sorted_k_smoothed_bigram_probability = second_step(text)


    sample_sentence_1 = "İlk elektrikli bilgisayar çalışmakta."
    sample_sentence_2 =  "Prof. Dr. Ahmet Kovalı'nın sayesinde bunu başarabildik."

    prob_of_sentence_1 = third_step(sample_sentence_1, text)
    prob_of_sentence_2 = third_step(sample_sentence_2, text)
    print(prob_of_sentence_1)
    file_name = "hw01_bilgisayar_Result.txt"
    with open(file_name, "w", encoding="utf-8") as f:

        f.write(f"Number of Sentences in File: {sentence_count}\n")

        f.write(f"Number of Total Tokens (Corpus Size): {token_count}\n")

        f.write(f"Number of Unique Words (Vocabulary Size): {vocab_size}\n\n")

        f.write("Unigrams Sorted wrt Frequencies (from Higher to Lower Frequencies):\n")
        for unigram, values in unigram_prob.items():
            count = values[0]
            prob = values[1]
            f.write(f"{unigram}: ( {count},   {prob} )\n")

        f.write("\nBigrams Sorted wrt Frequencies (from Higher to Lower Frequencies):\n")
        for bigram, values in bigram_prob.items():
            count = values[0]
            prob = values[1]
            f.write(f"{bigram}: ( {count},   {prob} )\n")

        f.write("\nTop 100 Bigrams wrt SmoothedProbability (from Higher to Lower):\n")
        for bigram, (count, smoothed_prob, prob) in top_100_sorted_k_smoothed_bigram_probability.items():
            f.write(f"{bigram}: ( {smoothed_prob},   {prob} )\n")

        f.write(f"\n{sample_sentence_1}     {prob_of_sentence_1}\n")

        f.write(f"\n{sample_sentence_2}     {prob_of_sentence_2}\n")
