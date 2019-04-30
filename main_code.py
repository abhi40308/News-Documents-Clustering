import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
# from bs4 import BeautifulSoup as Soup
import json

def parseLog(file):
    file = sys.argv[1]
    content = []
    with open(file) as f:
        content = f.readlines()
    content = [json.loads(x.strip()) for x in content]
    # print(content)
    
    data = json.loads(json.dumps(content))
    k=0
    # print(data)

# preprocessing ////////////////////////////////

    content_list = []
    for i in data:
        string_content = ""
        if "contents" in i:
            for all in i["contents"]:
                if "content" in all:
                    # print(str(all["content"]))
                    sample_str = str(all["content"])

                    new_str = ""
                    flag = 0
                    for i in sample_str:
                        if i == "<":
                            flag = 1
                        if i == ">":
                            flag = 0
                        if flag == 0:
                            new_str += i

                    string_content = string_content + new_str
            content_list.append(string_content)
    
    for i in range(15):
    	print(content_list[i])
    	print('\n\n')
    print('\n\n')

    news_df = pd.DataFrame({'document':content_list})

    # removing everything except alphabets`
    news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")

    # removing null fields
    news_df = news_df[news_df['clean_doc'].notnull()]
    # removing short words
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

    # make all text lowercase
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

    stop_words = stopwords.words('english')
    stop_words.extend(['span','class','spacing','href','html','http','title', 'stats', 'washingtonpost'])

    # tokenization
    tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())

    # remove stop-words
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    # print(tokenized_doc)

    # de-tokenization
    detokenized_doc = []
    for i in range(len(tokenized_doc)):
        if i in tokenized_doc:
            t = ' '.join(tokenized_doc[i])
            detokenized_doc.append(t)

    # print(detokenized_doc)


# tfidf ////////////////////////////////////////////

    from sklearn.feature_extraction.text import TfidfVectorizer

    # tfidf vectorizer of scikit learn
    vectorizer = TfidfVectorizer(stop_words=stop_words,max_features=10000, max_df = 0.5,
                                    use_idf = True,
                                    ngram_range=(1,3))

    X = vectorizer.fit_transform(detokenized_doc)

    # print(X.shape) # check shape of the document-term matrix

    terms = vectorizer.get_feature_names()
    # print(terms)

# applying k means //////////////////////////

    from sklearn.cluster import KMeans

    num_clusters = 5

    km = KMeans(n_clusters=num_clusters)

    km.fit(X)

    clusters = km.labels_.tolist()

    for i in range(15):
        print(clusters[i], end=" ")
    print('\n\n')


# applying lsa //////////////////////////////

    from sklearn.decomposition import TruncatedSVD
    from sklearn.utils.extmath import randomized_svd

    U, Sigma, VT = randomized_svd(X, 
                                  n_components=5,
                                  n_iter=100,
                              random_state=123)

    # print(U)

    # SVD represent documents and terms in vectors
    # svd_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=100, random_state=122)

    # svd_model.fit(X)
    
    # print(U.shape)

    for i, comp in enumerate(VT):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        print("Concept "+str(i)+": ")
        for t in sorted_terms:
            print(t[0])
        print(" ")

# topic visulation //////////////////////////////

    import umap

    print('\n\n')
    X_topics=U*Sigma
    # print(X_topics)
    embedding = umap.UMAP(n_neighbors=25, min_dist=0.5, random_state=12).fit_transform(X_topics)

    #printing points on the plot to compare
    print(embedding.shape)

    print(embedding[:15,0])
    print(embedding[:15,1])

    plt.figure(figsize=(7,5))
    plt.scatter(embedding[:, 0], embedding[:, 1], 
    c = clusters,
    s = 15, # size
    edgecolor='none'
    )
    plt.show()


if __name__ == "__main__":
    parseLog(sys.argv[1])