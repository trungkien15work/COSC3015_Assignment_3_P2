import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

vocab_des = {}
with open('vocab.txt', 'r') as file:
    for line in file:
        word, idx = line.strip().split(':')
        vocab_des[word] = int(idx)
tfidf_vec_des = TfidfVectorizer(vocabulary=vocab_des)
        
with open(r'.\stopwords_en.txt', 'r') as file:
    stopwords = file.read().splitlines()

def genVec(wv, tk_txts):
    # Fit and transform descriptions using the TfidfVectorizer
    X_tfidf_des = tfidf_vec_des.fit_transform(tk_txts)
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # if we haven't pre-processed the articles, it's a good idea to remove stop words

    for i in range(0,len(tk_txts)):
        tokens = tk_txts[i]
        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                temp = temp._append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        
        doc_vector = temp.sum(axis=1) # take the sum of each column
        docs_vectors = docs_vectors._append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    docs_vectors = docs_vectors.dropna()
    return docs_vectors

def tokenizeTxt(txt):
    tokens = []
    tokens_lower = []
    tokens_length = []
    tokens_more_than_1 = []
    final_list = []
    tokens_without_stopwords = []
    term_frequency = defaultdict(int)
    document_frequency = defaultdict(int) # Create a dictonary that contain the document frequency
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    tokens.append(tokenizer.tokenize(txt)) # Tokenize that description

    for tokene in tokens: # For every token array_list in the tokens
        token_list = [] # Create array token_list to store all tokens of 1 list
        for token in tokene: # For every token in the token array_list 
            token_list.append(token.lower()) # Lower the token and put it back to the token_list
        tokens_lower.append(token_list) # Get all the token_list into collection of token_lower
    
    for tokens in tokens:
        token_list = []
        for token in tokens:
            if len(token) >= 2: # If length of the word high than 2, keep the word
                token_list.append(token)
        tokens_length.append(token_list)
        
    for tokens in tokens_length:
        token_list = []
        for token in tokens:
            if token not in stopwords: # If the word not in stopwords list, keep the word
                token_list.append(token)
        tokens_without_stopwords.append(token_list)  
        
    for tokens in tokens_without_stopwords:
        for token in tokens: 
            term_frequency[token] += 1 # +1 for the word if it appear
 
    for tokens in tokens_without_stopwords:
        tokens_filtered_freq = []
        for token in tokens:
            if term_frequency[token] > 1: # If the term_frequency higher than 1, keep the word
                tokens_filtered_freq.append(token)
        tokens_more_than_1.append(tokens_filtered_freq)
        
    
    for document in tokens_more_than_1:
        unique_document_word = set(document) # Get all the word in a document unique
        for word in unique_document_word:
            document_frequency[word] += 1 # +1 for everytime the word appear in 1 document

    more_than_50 = []
    for word, count in document_frequency.items():
        more_than_50.append((word,count)) # Append the word and document_frequency count into more_than_50
        more_than_50.sort(key=lambda x: x[1], reverse=True) # Sort the more_than_50 from highest to lowest
        if len(more_than_50) > 50: # If this array length more than 50
            more_than_50.pop() # Remove the lowest count
    
    for tokens in tokens_more_than_1:
        token_list = []
        for token in tokens:
            if token not in more_than_50: # If the word not in more_than_50, keep the word
                token_list.append(token)
        final_list.append(token_list)
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = [[lemmatizer.lemmatize(token) for token in tokens] for tokens in final_list]
    result = []
    for word in lemmatized_list[0]:
        result.append(word)
    return result