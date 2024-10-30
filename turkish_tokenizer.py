import regex as re
from collections import Counter

class TurkishTokenizer:

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        target_characters = {'.', '!', '?'}

        #Abbreviations - This list can be expanded. This is just example.
        medical_abbreviations = ['dr', 'prof', 'doç']
        military_abbreviations = ['as', 'iz', 'asb', 'astsb', 'atğm', 'bçvş', 'bl', 'bnb',
                                  'çvş', 'dz', 'kuv', 'k', 'gen', 'gnkur', 'hv', 'kuv', 'svn',
                                  'ıs', 'ısth', 'kad', 'kd', 'kora', 'korg', 'kur', 'bşk', 'lv',
                                  'mu', 'nö', 'sb', 'onb', 'or', 'ora', 'ord', 'org', 'p', 'tb',
                                  'tğm', 'tnk', 'top', 'top', 'tug', 'tuğa', 'tüm', 'tüma', 'tümg',
                                  'ulş', 'uz', 'çvş', 'uzm', 'üçvş', 'ütğm', 'yb', 'yd', 'yzb']

        abbreviations = medical_abbreviations + military_abbreviations

        if text[-1] in target_characters:
            text = text[:-1] + ""

        for i in range (len(text)):
            if text[i] in target_characters:
                # checking if a number comes before the dot, also checks if the sentence ends with a number and ! or ?
                # this solution does not work for the sentences that and with a number and a dot. For example:
                # "Bu sorunun cevabı 7."
                # To solve this problem we would need to know the meaning of the number and why it is used
                # Is it seven? or seventh? The solution takes it as "seventh" and does not consider the probability of
                # it being "seven".
                # But for the given examples of hw01_tinytr.txt and hw01_bilgisayar.txt, it works just fine.
                if i > 0 and re.match(r"\d", text[i-1]) and (text[i] != "!" or text[i] == "?"):
                    continue

                # checking abbreviations
                for abbr in abbreviations:
                    if text[max(0, i - len(abbr) - 1):i].strip() == abbr:
                        break

                else:
                    text = text[:i] + "<\\s> <s>" + text[i + 1:]

        text = "<s> " + text.strip() + " <\\s>"
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def word_tokenizer(text):
        text = text.lower()
        punctuations = [
            '.', ',', '!', '?', ';', ':', '-', '_', '(', ')', '[', ']', '{', '}',
            '"', "'", '...', '–', '—', '/', '\\', '|', '@', '#', '$', '%', '^', '&',
            '*', '+', '=', '<', '>', '`', '~'
        ]
        tokens = re.findall(r"<s>|<\\s>|\w+|[^\w\s]", text, re.UNICODE)
        cleaned_tokens = [token for token in tokens if token not in punctuations]

        return cleaned_tokens
    @staticmethod
    def vocab_freq(text):
        tokens = re.findall(r"<s>|<\\s>|\w+", text, re.UNICODE)
        vocabulary = {}
        for word in tokens:
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1
        return vocabulary

    @staticmethod
    def vocab(text):
        tokens = re.findall(r"<s>|<\\s>|\w+", text, re.UNICODE)
        return set(tokens)

    @staticmethod
    def sentence_tokenizer(text):
        sentences = re.findall(r"<s>.*?<\\s>", text)
        return sentences

    @staticmethod
    def n_gram(text, n = 1):
        ngram_complete = []
        sentences = TurkishTokenizer.sentence_tokenizer(text)
        for sentence in sentences:
            tokens = TurkishTokenizer.word_tokenizer(sentence)
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngram_complete.append(ngram)
        return ngram_complete

    @staticmethod
    def sort_ngrams(ngram):
        ngram_frequencies = Counter(ngram)
        sorted_ngrams = ngram_frequencies.most_common()
        return sorted_ngrams

    @staticmethod
    def replace_unk(text):
        word_frequencies = TurkishTokenizer.vocab_freq(text)

        min_token = min(word_frequencies, key=word_frequencies.get)
        min_frequency = word_frequencies[min_token]

        text = text.replace(f" {min_token} ", " UNK ")

        word_frequencies['UNK'] = word_frequencies.get('UNK', 0) + min_frequency
        del word_frequencies[min_token]
        return text


    @staticmethod
    def ngram_probabilities(text, add_k_smoothing = True, k = 1, n = 1):
        ngrams = TurkishTokenizer.n_gram(text, n)
        sorted_ngrams = TurkishTokenizer.sort_ngrams(ngrams)
        vocab_size = len(TurkishTokenizer.vocab(text))
        tokens = TurkishTokenizer.word_tokenizer(text)
        total_words = len(tokens)
        if n > 1:
            context_counts = Counter(TurkishTokenizer.n_gram(text, n - 1))
        else:
            context_counts = Counter(ngrams)

        ngram_probs = {}
        for ngram, count in sorted_ngrams:
            context = ngram[:n - 1] if n > 1 else ngram
            if n == 1:
                probability = count / total_words
                k_smoothed_probability = (count + k) / (total_words + k * vocab_size)
            else:
                probability = count / context_counts[context]
                k_smoothed_probability = (count + k) / (context_counts[context] + k * vocab_size)
            if add_k_smoothing:
                ngram_probs[ngram] = (count, k_smoothed_probability, probability)
            else:
                ngram_probs[ngram] = (count, probability)

        ngram_probs[('unk', 'unk')] = (0, k/(k*vocab_size), 0)
        return ngram_probs

    @staticmethod
    def prob_of_a_given_corpus(text, ngram_prob, vocabulary, k=1, n=1):
        processed_text = TurkishTokenizer.preprocess_text(text)
        ngrams = TurkishTokenizer.n_gram(processed_text, n)
        sorted_ngrams = TurkishTokenizer.sort_ngrams(ngrams)
        vocab_size = len(TurkishTokenizer.vocab(text))

        if n > 1:
            context_counts = Counter(TurkishTokenizer.n_gram(text, n - 1))
        else:
            context_counts = Counter(ngrams)

        total_probability = 1.0

        for ngram in ngrams:
            probability = 1.0

            if ngram in ngram_prob:
                probability = ngram_prob[ngram][1]

            else:
                updated_ngram = tuple('unk' if word not in vocabulary else word for word in ngram)
                ngram_dict = dict(sorted_ngrams)

                if ngram in ngram_dict:
                    count = ngram_dict[ngram]
                else:
                    count = 0
                if updated_ngram[0] in vocabulary and updated_ngram[1] == 'unk':
                    context = updated_ngram[0]
                    probability = (count + k) / (context_counts[context] + k * vocab_size)

                elif updated_ngram[0] == 'unk' and updated_ngram[1] in vocabulary:
                    context = updated_ngram[1]
                    probability = (count + k) / (context_counts[context] + k * vocab_size)


            total_probability *= probability

        return total_probability