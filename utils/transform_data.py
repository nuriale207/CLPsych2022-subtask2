import multidict as multidict
import numpy as np

def ndarrayToString(predicciones):
    """ Recibe las predicciones en formato ndarray y las convierte en String

                 Parámetros:
                        predicciónes -- matriz ndarray con las predicciones
    """
    x=predicciones.shape[0]

    y=predicciones.shape[1]

    pred=""
    for i in range(0,x):
        fila=""

        for j in range(0,y):
            num=predicciones[i,j]
            fila=fila+str(num)+","
        fila=fila.rstrip(',')

        pred=pred+fila+"\n"

    return pred


def listOfListsToString(lista):
    """ Convierte una lista de listas en un String donde en cada fila se encuentra el contenido de cada lista separado por comas

                     Parámetros:
                            lista -- lista de listas que se convierte en String
        """

    texto=""
    for i in lista:
        fila=""
        for j in i:
            fila=fila+str(j)+", "

        texto=texto+fila+"\n"

    return texto

def stringToCorpus(corpus):
    """ Recibe un String y lo convierte en una lista de listas
                     Parámetros:
                            corpus -- String a convertir en lista de listas
        """

    i = 0
    documentos=corpus.split("\n")
    corpus = []

    for document in documentos:

        document = document.replace("\n", "")
        document = document.split(',')

        corpus.append(document)
        i = i + 1
    corpus.pop()
    return corpus

def lineToDict(line):
    """ Recibe un String que contiene un término y un valor y lo convierte en un diccionario
                         Parámetros:
                                line -- String que contiene palabras y su frecuencia separados por comas
            """
    corpus = []
    fullTermsDict = multidict.MultiDict()

    dic={}
    palFreq = line.split("\t")
    for text in palFreq:
        if text !="":
            text = text.split(",")
            palabra = text[0]

            frecuencia = text[1]

            fullTermsDict.add(palabra, float(frecuencia))
    return fullTermsDict

def textToTopicFreqDict(file):
    """ Recibe el path de un archivo y lo carga en un diccionario donde se almacenan los términos y su frecuencia
                         Parámetros:
                                file -- path al archivo
            """

    file = open(file)
    lista = []
    dicEtiquetasTopics = {}
    listaTodasPalabras = []
    # Obtener representaccion de topics y etiquetas asociadas
    for line in file:
        if line.__contains__(":"):
            etiqueta = line.split(":")[0]
            line = line.replace("\n", "")
            line = line.split(":")[1]
            line = line[1:]
            palabras = line.split("\t")

            todPalabras = ""
            for conjPalabras in palabras:
                palabra = conjPalabras.split(",")[0]
                if (palabra != ""):
                    listaTodasPalabras.append(palabra)
                    todPalabras = todPalabras + palabra + ","

            if todPalabras not in dicEtiquetasTopics.keys():
                lista.append(todPalabras)
                dicEtiquetasTopics[todPalabras] = etiqueta
            else:
                dicEtiquetasTopics[todPalabras] = etiqueta + "," + dicEtiquetasTopics[todPalabras]

    return dicEtiquetasTopics,listaTodasPalabras

def deMatrizSparseAMatrizEtiquetas(diccionarioEtiquetas,stringPredicciones):
    """ Recibe un array de predicciones en formato sparse y lo convierte en non sparse en base a un diccionario
                             Parámetros:
                                    diccionarioEtiquetas -- diccionario que en su i-ésima posición almacena el valor
                                                            correspondiente a la i-ésima etiqueta
                                    stringPredicciones -- string que en cada linea contiene 1 o 0 en base a si se ha predicho o no la etiqueta
                """
    numToTagAll = []
    stringPredicciones2 = stringPredicciones.split('\n')

    for line in stringPredicciones2:
        line = line.split(',')
        numToTagDoc = []
        i = 0
        for num in line:

            if num != '':
                if int(num) == 1:
                    numToTagDoc.append(diccionarioEtiquetas[i])
            i = i + 1

        numToTagAll.append(numToTagDoc)

    return numToTagAll


# def generarDiccionario(corpus):
#     print("Generando diccionario...")
#     diccionario= {}
#     diccionarioInverso={}
#     setPal=set()
#     i=0
#     lista=""
#     for document in corpus:
#         for word in document:
#             if word not in setPal:
#                 setPal.add(word)
#                 diccionario[word]=i
#                 diccionarioInverso[i]=word
#                 i+=1
#                 lista+=word+"\n"
#
#     return diccionario,lista,diccionarioInverso
#
# def bow(corpus):
#     diccionario,dicTxt,diccionarioInverso=generarDiccionario(corpus)
#     bow=[]
#     bowText=",counts\n"
#     i=0
#     bowTokens=[]
#     bowTokensText=",tokens\n"
#     for document in corpus:
#         palabras=list(set(document))
#         tokens=list(dict(Counter(document)).values())
#         lista=[]
#         for palabra in palabras:
#             lista.append(diccionario[palabra])
#         bowTokens.append(lista)
#         bow.append(tokens)
#         bowTokensText+=str(i)+","+str(lista)+"\n"
#         bowText+=str(i)+","+str(tokens)+"\n"
#         i+=1
#     return bowText,bow,bowTokens,bowTokensText,diccionarioInverso,dicTxt

# def bow(corpus,diccionario):
#     bow = []
#     bowText = ",counts\n"
#     i = 0
#     bowTokens = []
#     bowTokensText = ",tokens\n"
#     for document in corpus:
#         palabras = list(set(document))
#         palabrasValidas=set()
#         tokens = []
#         lista = []
#         for palabra in palabras:
#             if palabra in diccionario.keys():
#                 lista.append(diccionario[palabra])
#                 palabrasValidas.add(palabra)
#
#         bowTokens.append(lista)
#         for palabra in palabras:
#             if palabra in palabrasValidas:
#                 tokens.append(palabra)
#
#         tokens = [list(dict(Counter(tokens)).values())]
#         bow.append(tokens)
#         bowTokensText += str(i) + "," + str(lista) + "\n"
#         bowText += str(i) + "," + str(tokens) + "\n"
#         i += 1
#     return bowText,bow,bowTokens,bowTokensText

# def bow2(corpus,diccionario):
#     vectorizer=CountVectorizer(vocabulary=diccionario, max_df=0.7,min_df=10)
#     X=vectorizer.fit_transform(corpus)
#     return X




def crearMatrizEtiquetas(topic_label_dict, etiquetasTrain):
    """ Crea una matriz sparse de ceros y unos partiendo de un diccionario

                  Parámetros:
                  topic_label_dict -- diccionario que en la i-ésima posición contiene el i-ésimo topic.
                  etiquetasTrain -- lista de listas donde se almacenan las etiquetas por su nombre

              """
    docKop = len(etiquetasTrain)
    tagKop = len(topic_label_dict)
    #tagIndex=0
    sparse = np.zeros((docKop, tagKop))

    for docIndex, document in enumerate(etiquetasTrain):

        if docIndex % 1000 == 0:
            print("Procesando documento: "+str(docIndex))

        for tag in document:
            for index, element in enumerate(topic_label_dict):
                if element == tag:
                    tagIndex = index

            sparse[docIndex, tagIndex] = 1

    return sparse


def crearDiccionarioEtiquetas(matrizEtiquetas):
    """ Crea un diccionario que relaciona la posición de las etiquetas en la matriz sparse con el nombre de las etiquetas

                     Parámetros:
                     matrizEtiquetas -- lista de listas en la que en cada lista contiene las etiquetas de un documento
                                        almacenadas por su nombre
                     Devuelve:
                        diccionario:diccionario que relaciona la posición de las etiquetas en la matriz sparse con el nombre de las etiquetas
                        listaNombres: lista que ordena las etiquetas según están situadas en la matriz sparse

    """
    listaNombres=[]

    for documento in matrizEtiquetas:
        #linea=documento.rstrip("\n")
        #etiquetas=linea.split(",")
        for etiqueta in documento:
            if(etiqueta not in listaNombres and etiqueta!="" ):
                listaNombres.append(etiqueta)
    indices=list(range(len(listaNombres)))

    diccionario = dict(zip(listaNombres,indices))
    return diccionario,listaNombres