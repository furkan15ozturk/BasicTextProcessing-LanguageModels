from audioop import avgpp

import regex as re

class TurkishTokenizer:

    def preprocessText(self, text):
        text = text.lower()
        target_characters = {'.', '!', '?'}
        abbreviations = ['Dr', 'Prof']
        for i in range (len(text)):
            number_pattern = r"\d\."
            if text[i] in target_characters:
                if i > 0 and re.match(r"\d", text[i-1]):
                    continue
                # checking abbreviations
                for abbr in abbreviations:
                    if text[max(0, i - len(abbr) - 1):i].strip() == abbr:
                        break
                else:
                    text = text[:i] + "<\\s>" + text[i + 1:]
        text = re.sub(r"\s+", " ", text)
        return text
    
    def tokenize(self, text):
        text = text.lower()

        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

        return tokens
    
    def vocabFreq(self, text):
        tokens = re.findall(r"\w+", text, re.UNICODE)
        vocabulary = {}
        for word in tokens:
            if word in vocabulary:
                vocabulary[word] += 1
            else: 
                vocabulary[word] = 1
        return vocabulary