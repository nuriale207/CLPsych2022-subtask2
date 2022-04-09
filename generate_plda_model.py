import argparse
import json
from pathlib import Path

import pandas as pd

import Graphics
import TransformData
from data_reader import json_reader, all_csv_reader, process_data, user_csv_reader
from generate_topics import createPLDA, obtenerVectorTopics
from preprocess_data import vocab_size


def create_arguments():
    parser = argparse.ArgumentParser(
        description='This runnable preproccesses the documents and generates a json file for each user with the '
                    'predicted topics in each post. It also generates a json file with the most important words in each '
                    'topic and their weight.')

    parser.add_argument('--json', dest='json_file', nargs='+',
                        help='Path to the json file with the users information.')
    parser.add_argument('--timelines', dest='time_dir', nargs='+',
                        help='Path to the directory with the timelines tsv files.')
    parser.add_argument('-k', dest='topics_per_label', action='store', default=1,
                        help='Indicates the amount of topics per label to generate')
    parser.add_argument('-n', dest='topic_words',  nargs='+', default=1,
                        help='Indicates the amount of words to include in the topics information file')
    parser.add_argument('--path',dest='path_dir', nargs='+',
                        help='Path to the directory to save the generated plda model.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_arguments()
    user_dir = args.json_file[0]
    time_dir = args.time_dir[0]
    num_topics=args.topics_per_label[0]
    dest_dir=args.path_dir[0]
    n=args.topic_words[0]

    users_json=json_reader(user_dir)
    all_users_timelines=all_csv_reader(users_json,time_dir)


    all_users_timelines,preprocessed_docs=process_data(all_users_timelines)
    labels=list(all_users_timelines["label"])


    vocab,vocab_n=vocab_size(preprocessed_docs)

    alpha=50/int(num_topics)*(len(set(labels)))
    eta=200/vocab_n


    plda_model=createPLDA(0,0,2,alpha,eta,preprocessed_docs,labels)

    all_users_topics=dict()
    for user in users_json:
        df=user_csv_reader(user,users_json,time_dir)
        df,preprocessed_docs=process_data(df)
        lista_topics=[]
        for i in range(len(preprocessed_docs)):
            topics=obtenerVectorTopics(plda_model,preprocessed_docs[i])
            lista_topics.append(list(topics))
            # lista_topics.append({"post_id":df["postid"][i],"topics":list(topics)})
        df_topics=pd.DataFrame(data={"postid":df["postid"],"date":df["date"],"label":df["label"],"topics":lista_topics})
        filepath = Path(dest_dir+'/'+user+"_topics.tsv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df_topics.to_csv(filepath,sep='\t')

    plda_model.save(dest_dir+'/plda_model.pkl')

    infoTopics = ""
    j = 0
    for i in range(len(plda_model.topic_label_dict)):
        l = 0
        while (l < plda_model.topics_per_label and j < plda_model.k):
            infoTopics = infoTopics + plda_model.topic_label_dict[i] + ": "
            tuplas = plda_model.get_topic_words(j, int(n))

            for k in range(len(tuplas)):
                tupla = tuplas[k]
                palabra = tupla[0]
                probabilidad = str(tupla[1])
                infoTopics = infoTopics + palabra + "," + probabilidad + "\t"
            infoTopics = infoTopics + "\n"
            j += 1
            l += 1
    # FileHandler.guardarDocumento(infoTopics, "../Archivos/infoTopics.txt")
    with open(dest_dir + '/topic_words.txt', 'w') as outfile:
        outfile.write(infoTopics)

    topics = infoTopics.split("\n")

    i = 0
    for topic in topics:
        dic = {}

        if topic == "":
            break

        if topic.__contains__(':'):
            topic = topic.split(':')[1]

        dic = TransformData.lineToDict(topic)
        # Graphics.crearNubesPalabras(dic,True,"../Archivos/nubes/"+str(i))
        Graphics.crearNubesPalabras(dic, True, dest_dir + "/" + str(i))
        i = i + 1