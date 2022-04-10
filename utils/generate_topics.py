import tomotopy
import tomotopy as tp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def LDA(p_min_cf, p_min_df, p_rm_top, p_k, p_alpha, p_eta,p_corpus):
    """ Devuelve una matriz de los vectores de probabilidades de los topics y el modelo generado
                                        Parámetros:
                                            p_min_cf -- mínima frecuencia de palabras en el modelo
                                            p_min_df -- mínima frecuencia de palabras en el documento
                                            p_rm_top -- cantidad de palabras comunes a eliminar del modelo
                                            p_k -- cantidad de topics a generar
                                            p_alpha -- representa la densidad documento-tema
                                            p_eta -- representa la densidad tema-palabra
                                            p_corpus -- una lista de documentos que se agregarán al modelo
                            """

    ldaModel=tomotopy.LDAModel(tw=tomotopy.TermWeight.IDF, min_cf=p_min_cf, min_df=p_min_df, rm_top=p_rm_top, k=p_k, alpha=p_alpha, eta=p_eta)

    for document in p_corpus:
        ldaModel.add_doc(document)
    ldaModel.train(500)
    lista = []
    i = 0
    for index, document in enumerate(p_corpus):
        vector=obtenerVectorTopics(ldaModel,document)
        lista.append(vector)
        i = i + 1
        if (i % 5 == 0):
            print("Modelo creado hasta documento: " + str(i))
    return lista,ldaModel


def obtenerVectorTopics(modelo,documento):
    """ Devuelve un vector con la probabilidad de los tópicos generados en el modelo pasado por parámetro
        del documento nuevo.
                                        Parámetros:
                                            modelo -- modelo LDA generado
                                            documento -- nuevo documento del que sacar los topics
                            """

    doc = modelo.make_doc(documento)

    infer = modelo.infer(doc, iter=100, tolerance=-1, workers=0, parallel=0, together=False)
    return infer[0]

def crearModeloPLDA( parametros, cuerpo, etiquetas):
    """ Crea el modelo PLDA, dados unos parámetros, documentos y etiquetas

                     Parámetros:
                            parametros -- parámetros con los que crear el modelo
                            cuerpo -- lista de listas con las palabras de cada documento
                            etiquetas -- lista de listas con las etiquetas de cada documento
    """

    return createPLDAlist(parametros, cuerpo, etiquetas)



def createPLDAlist(parametros, cuerpo, etiquetas):
    """ Crea el modelo PLDA, dados unos parámetros, documentos y etiquetas

                         Parámetros:
                                parametros -- parámetros con los que crear el modelo
                         min_cf,rm_top,latent_topics,topics_per_label,alpha,eta   \n"
              "             Por ejemplo: 10,0,0,1,0.2,0.1       cuerpo -- lista de listas con las palabras de cada documento
                                etiquetas -- lista de listas con las etiquetas de cada documento
    """
    min_cf = parametros[0]
    rm_top = parametros[1]
    latent_topics = parametros[2]
    topics_per_label = parametros[3]
    alpha = parametros[4]
    eta = parametros[5]

    return createPLDA(min_cf, rm_top, latent_topics, topics_per_label, alpha, eta,cuerpo,etiquetas)


def createPLDA(min_cf, rm_top, topics_per_label, alpha, eta, cuerpo, etiquetas):
    """ Crea el modelo PLDA, dados unos parámetros, documentos y etiquetas

                             Parámetros:
                                     min_cf -- mínima frecuencia de palabras en el modelo
                                     rm_top -- cantidad de palabras comunes a eliminar del modelo
                                     latent_topics -- cantidad de topics a generar
                                     topics_per_label -- cantidad de topics a generar por etiqueta
                                     alpha -- representa la densidad documento-tema
                                     eta -- representa la densidad tema-palabra
                                    cuerpo -- lista de listas con las palabras de cada documento
                                    etiquetas -- lista de listas con las etiquetas de cada documento
    """
    model = tp.PLDAModel(tw=tp.TermWeight.IDF, min_cf=min_cf, rm_top=rm_top,
                         topics_per_label=topics_per_label, alpha=alpha, eta=eta)
    count = 0

    for document, tags in zip(cuerpo, etiquetas):
        model.add_doc(document, labels=[tags])
        count += 1
    model.train(500)
    return model


def createLDAlist(parametros, cuerpo):
    """ Crea el modelo LDA, dados unos parámetros, documentos y etiquetas

                             Parámetros:
                                    parametros -- parámetros con los que crear el modelo
                                    cuerpo -- lista de listas con las palabras de cada documento
                                    etiquetas -- lista de listas con las etiquetas de cada documento
        """
    min_cf = parametros[0]
    min_df = parametros[1]
    rm_top = parametros[2]
    num_topics = parametros[3]
    alpha = parametros[4]
    eta = parametros[5]

    return LDA(min_cf, min_df,rm_top, num_topics, alpha, eta)




def load_PLDA_model(path):
    """ Carga el modelo PLDA que estaba almacenado en un path
                   Parámetros:
                      path -- path en el que estaba almacenado el modelo
        """
    model = tp.PLDAModel.load(path)
    return model