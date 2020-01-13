import re
import unicodedata
from bs4 import BeautifulSoup
import numpy as np

characters = ":_-’'!?., abcdefghijklmnopqrstuvwxyzáàâãäåæçéèêëíìîïðñòóôõöøùúûüýþÿ"
letters = "-' abcdefghijklmnopqrstuvwxyzáàâãäåæçéèêëíìîïðñòóôõöøùúûüýþÿ"

def check_alphanumeric(character):
    encoded = character.encode('utf-8')
    return str(encoded) != ("b" + "'" + character + "'")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def remove_special_characters(string):
    normalized = unicodedata.normalize('NFKC', string)
    # TODO better handle numbers
    processed = [char for char in normalized if check_alphanumeric(char) is False or (check_alphanumeric(char) and char in characters) or char == '\n']
    return ''.join(processed)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def clean_text( doc ):

    isHtml = re.match(r'<\w+>', doc)
    if isHtml and isHtml.group():
        doc = BeautifulSoup( doc, features="html.parser").get_text()

    doc = remove_special_characters(doc)

    doc = doc.replace(' undefined ', '')
    doc = doc.replace(' null ', '')
    doc = doc.replace(':NOT_FROM_RESUME:', '')

    # clean up newlines/carriage returns
    doc = doc.replace('\r\n', '\n')
    doc = re.sub(r'\n\n+', '\n\n', doc)

    # replace and deal with punctuation
    doc = re.sub(r'(-|_|:)', ' ', doc)  # not certain this is needed any more
    doc = re.sub(r',|\/|\(|\)|\&', ' ', doc)  # not certain this is needed any more
    doc = re.sub(r'\.|!|\?|,', '', doc)  # maybe do this based on a flag, since we might want to keep for structure.

    # split consecutive words where the first is lowercase, and the second is upper case
    doc = re.sub(r"(?<![A-Z])(?<!^)([A-Z])", r" \1", doc)

    # collapse spaces
    doc = doc.replace('\\t', ' ')
    doc = re.sub(r'  +', ' ', doc)

    # custom stuff
    doc = re.sub(r'http\S+', ' ', doc ) # remove URLs
    doc = re.sub("\S*@\S*\s?", ' ', doc ) # remove email addresses
    doc = doc.replace('\n\n', ' ')
    doc = doc.replace('\n', ' ')
    doc = doc.replace(u'\xa0', u' ')
    doc = doc.replace( '\'s', '')

    #doc = replace_numbers(doc)
    doc = re.sub(r'[0-9]+', '', doc)

    # remove punctuation, special characters
    to_remove = [ '\'', '"', ':', ';', '>', '<', ',', '-','^']
    for c in to_remove:
        doc = doc.replace( c, ' ' )

    return doc

#########################################

def normalize_text_nltk( text, min_length=3, language="english" ):
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import re
    from nltk.corpus import stopwords

    lemmatizer = WordNetLemmatizer()

    cachedStopWords = stopwords.words(language)

    text = re.sub("[^a-zA-Z]", " ", text)
    text = word_tokenize(text)
    text = [ t.lower() for t in text]
    text = [ t for t in text if len(t)>=min_length]
    text = [t for t in text if t not in cachedStopWords]
    text = [lemmatizer.lemmatize(t, pos='v') for t in text] # 'v' = verbs

    text = " ".join( [w for w in text] )

    return text

#########################################
