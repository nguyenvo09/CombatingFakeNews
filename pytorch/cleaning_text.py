import re
from nltk.tokenize import TweetTokenizer

def remove_special_tokens(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("\xa0", " ")
    text = text.replace('’', "'")
    text = text.replace("”", "\"")
    text = text.replace("“", "\"")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    args = text.split()
    text = " ".join(args)
    return text

def clean_dataset(line):
    line = line.replace("\n", "")
    line = re.sub(r"https?:\S+", "url", line)
    line = remove_special_tokens(line)

    tknzr = TweetTokenizer()
    line = tknzr.tokenize(line)
    line = " ".join(line)

    tokens = line.split()
    for idx, token in enumerate(tokens):
        if token.startswith("@"):
            tokens[idx] = "@user"
    line = " ".join(tokens)

    return line