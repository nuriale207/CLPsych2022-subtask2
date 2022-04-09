import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud



def crearNubesPalabras(diccionario, guardar, path):
    """ Genera una imagen png en la que se muestran las palabras de los tópicos

                         Parámetros:
                                diccionario -- diccionario que contiene los términos del tópico junto a su porcentaje de aparición
                                guardar -- booleano que indica si hay que generar el gráfico en un path en concreto
                                path -- string con la ruta en la que almacenar el gráfico generado
            """
    #wc = WordCloud(width=2133,height=1067,background_color="black", max_words=6000)
    #wc = WordCloud(width=1030,colormap="tab10", height=536,  max_words=6000,background_color="white")
    wc = WordCloud(width=515,height=268,max_words=10000, background_color="white")

    #wc = WordCloud(width=1400, height=2234, colormap="tab10", background_color="white", mode="RGBA",
     #              max_words=6000)
    # makes the circle using numpy
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)
    # Generate a word cloud image
    wc.generate_from_frequencies(diccionario)

    # Display the generated image:
    # the matplotlib way:

    #plt.imshow(wc, interpolation='bilinear')
    plt.imshow(wc, interpolation=None)

    plt.axis("off")

    # lower max_font_size
    plt.axis("off")

    if guardar:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
