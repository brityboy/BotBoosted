import numpy as np
import pandas as pd
import twokenize as tw
import re
import string
from nltk import regexp_tokenize


def fix_the_sequence_of_repeated_characters(word):
    '''
    INPUT
         - word: this is the word being inspected for multiple letters
    OUTPUT
         - str
    Returns a cleaned up string where the sequence of letters repeating
    more than a few times have been removed
    '''
    word = word.lower()
    letters = set(word)
    for letter in letters:
        strings = re.findall(letter+'{3,}', word)
        strings.sort(key=len, reverse=True)
        if strings:
            for string in strings:
                word = word.replace(string, letter*2)
    return word


def tokenize_tweet(text):
    '''
    INPUT
         - text string for the tweettes
    OUTPUT
         - list
    Returns a list of the different tokens inside the tweet
    WHERE the ff things have been DONE which are twitter specific
     - hashtags are removed (words beginning with #)
     - mentions (words beginning with @) are removed
     - url's are replaced with the word 'url'
    '''
    token_list = []
    text = text.replace('\n', ' ')
    for token in text.split():
        url = tw.Url_RE.search(token)
        mentions = tw.Entity_RE.search(token)
        time = tw.Time_RE.search(token)
        numnum = tw.NumNum_RE.search(token)
        numcomma = tw.NumberWithCommas_RE.search(token)
        separators = tw.Separators_RE.search(token)
        emoticon = tw.emoticons.Emoticon_RE.search(token)
        if url:
            token_list.append('url')
        elif token[0] == '#':
            pass
        elif time:
            pass
        elif separators:
            pass
        elif numcomma:
            pass
        elif numnum:
            pass
        elif token[0] == '@':
            pass
        elif mentions:
            pass
        elif token == 'RT':
            pass
        elif token == 'Retweeted':
            pass
        elif type(token) == int:
            pass
        elif emoticon:
            token_list.append(tw.emoticons.analyze_tweet(token))
        else:
            token = token.translate(None, string.punctuation)
            token = fix_the_sequence_of_repeated_characters(token)
            token_list.append(token)
    return ' '.join(token_list)




if __name__ == "__main__":
    df = pd.read_csv('data/clintontweets.csv')
    documents = [tokenize_tweet(document) for document in df.text.values if type(document) == str]
    text = df.text.values[49723]
    # text = documents[32]
    # for i, document in enumerate(documents):
    #     print(i, tokenize_tweet(document))
    # print(fix_the_sequence_of_repeated_characters("TTTCCGACTTTTTGACTTACGAAAAAA"))
    # print tw.emoticons.analyze_tweet(':)')
