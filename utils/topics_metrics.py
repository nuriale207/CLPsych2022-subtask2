from collections import Counter

import nltk
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import jensenshannon
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def distanciaEuclidea(vector1,vector2):
    """ Calcula la distancia Euclidea entre dos vectores
                            Parámetros:
                                vector1 -- vector de datos
                                vector2 -- vector de datos
                        """
    return distance.euclidean(vector1,vector2)

def distanciaManhattan(vector1,vector2):
    """ Calcula la distancia Manhattan entre dos vectores
                            Parámetros:
                                vector1 -- vector de datos
                                vector2 -- vector de datos
                        """
    return distance.minkowski(vector1,vector2,1)

def jensenShannonDistance(v1,v2):
    """ Calcula la distancia Jensen shannon entre dos vectores
                                Parámetros:
                                    vector1 -- vector de datos
                                    vector2 -- vector de datos
                            """
    return jensenshannon(v1,v2)

def get_topic_diversity(modelo,numTopics,top_n):
    """ Calcula la diversidad de tópicos de un modelo

                         Parámetros:
                                modelo -- modelo de tomotopy
                                numTopics -- cantidad de tópicos con los que se ha generado el modelo o
                                            la cantidad que se quiere tener en cuenta
                                top_n -- cantidad de palabras más relevantes de los tópicos que se quieren
                                        tener en cuenta para calcular la diversidad
            """
    wordDict={}
    for i in range(numTopics):
        topWords=modelo.get_topic_words(i,top_n=top_n)
        for word in topWords:
            if word[0] in wordDict.keys():
                wordDict[word[0]]+=1
            else:
                wordDict[word[0]] = 1

    dictSorted={k: v for k, v in sorted(wordDict.items(), key=lambda item: item[1])}
    recuento=dict(Counter(list(dictSorted.values())))
    numUnique=recuento[1]



    return numUnique/(numTopics*25)


def get_document_frequency(data, wi, wj=None):
    """ Calcula la frecuencia de que los términos wi y wj coaparezcan en un documento

                         Parámetros:
                                data -- lista de listas con los documentos
                                wi -- primer término
                                wj -- segundo término que se comprueba si aparece junto al primero
            """

    D_wj = 0
    D_wi_wj = 0
    for doc in data:

        if wj in doc:
            if wi in doc:
                D_wi_wj += 1
    return D_wi_wj

def get_topics_coherence(vectores, data, modelo):
    """ Calcula la coherencia de tópicos de un conjunto

                             Parámetros:
                                    vectores -- lista (matriz) con los vectores de tópicos
                                    data -- lista de listas con los documentos
                                    modelo -- modelo de tópicos de tomotopy
                """
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(vectores[0])
    vocab=modelo.vocabs
    for k in range(num_topics):
        TC_k,counter=get_topic_coherence(data,modelo,k)
        TC.append(TC_k)
    print('counter: ', counter)
    print('num topics: ', len(TC))
    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))
    return TC

def get_topic_coherence(data,modelo,k):
    """ Calcula la coherencia del tópico k

                             Parámetros:
                                    data -- lista de listas con los documentos
                                    modelo -- modelo de tópicos de tomotopy
                                    k -- índice del tópico del que calcular la coherencia
                """

    top_10 = modelo.get_topic_words(k, top_n=10)
    top_words = dict(top_10).keys()
    TC_k = 0
    counter = 0
    D = len(data)
    for i, word in enumerate(top_10):
        # get D(w_i)

        D_wi = modelo.vocab_df[list(modelo.vocabs).index(word[0])]
        j = i + 1
        tmp = 0
        while j < len(top_10) and j > i:
            # get D(w_j) and D(w_i, w_j)
            D_wj = modelo.vocab_df[list(modelo.vocabs).index(top_10[j][0])]
            D_wi_wj = get_document_frequency(data, word[0], top_10[j][0])
            # get f(w_i, w_j)
            if D_wi_wj == 0:
                f_wi_wj = -1
            else:
                f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
            # update tmp:
            tmp += f_wi_wj
            j += 1
            counter += 1
        # update TC_k
        TC_k += tmp
    return TC_k,counter


def sentiment_score(tuples,polarity=False):
    """
    Computes a score from 0 to 1 given a set of tuples with the words in a topic.
    Each word gets a value of positivity, negativity or neutral. Positive words are balanced with a 0, neutral ones with
    5 and positive ones with 10.
    :param tuples: a tuple with the form (word,weight) of the n-most salient words in a topic
    """
    sia = SentimentIntensityAnalyzer()
    topic_score=0
    weights={'neg': 0.0, 'neu': 0.5, 'pos': 1.0}
    for k in range(len(tuples)):
        tuple = tuples[k]
        word = tuple[0]
        probability = str(tuple[1])
        if(polarity==False):
            score=sia.polarity_scores(word)["compound"]
        else:
            scores=sia.polarity_scores(word)
            if(scores["neg"]>scores["neu"] & scores["neg"]>scores["pos"]):
                score=weights["neg"]
            elif(scores["pos"]>scores["neu"] & scores["pos"]>scores["neg"]):
                score = weights["pos"]
            else:
                score=weights["neu"]
        topic_score+=score

    return topic_score/len(tuples)
