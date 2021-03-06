import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from utils.data_reader import json_reader, all_csv_reader, csv_reader, process_data, user_csv_reader, concat_csv_reader
from utils.classification_utils import *
from utils.generate_topics import createPLDA, obtenerVectorTopics
from utils.preprocess_data import vocab_size
from utils.topics_metrics import sentiment_score
from sklearn.ensemble import RandomForestClassifier

def create_arguments():
    parser = argparse.ArgumentParser(
        description='This runnable preproccesses the documents and generates a json file for each user with the '
                    'predicted topics in each post. It also generates a json file with the most important words in each '
                    'topic and their weight.')

    parser.add_argument('--json', dest='json_file', nargs='+',
                        help='Path to the json file with the users information.')
    parser.add_argument('--timelines', dest='time_dir', nargs='+',
                        help='Path to the directory with the timelines tsv files.')
    parser.add_argument('--num_topics', dest='topics_per_label', action='store', default=1,
                        help='Indicates the amount of topics per label to generate')
    parser.add_argument('-n', dest='topic_words',  nargs='+', default=1,
                        help='Indicates the amount of words to include in the topics information file')
    parser.add_argument('-k', dest='folds',  nargs='+', default=1,
                        help='Indicates the amount of folds to make')
    parser.add_argument('--path',dest='path_dir', nargs='+',
                        help='Path to the directory to save the generated plda model and json files.')
    parser.add_argument('--polarity', dest='polarity', action='store_true', help="If set, each topic word is labelled"
                                                                                 "with 0 if it is negative, 0.5 if it is"
                                                                                 "neutral or 1 if it is positive. Otherwise"
                                                                                 "it takes into account the polarity level"
                                                                                 "of words")



    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_arguments()
    user_dir = args.json_file[0]
    time_dir = args.time_dir[0]
    num_topics=args.topics_per_label[0]
    num_topics=int(num_topics)
    dest_dir=args.path_dir[0]
    n=args.topic_words[0]
    k=args.folds[0]
    k=int(k)
    users_json = json_reader(user_dir)
    polarity=args.polarity
    # all_users_timelines = all_csv_reader(users_json, time_dir)

    n_users=len(users_json)

    indexes=get_random_indexes(n_users)

    users_per_fold=n_users//k

    first_index=0
    predictions_report=""
    all_macro=[]
    all_weighted=[]
    for fold in range(1,k+1):
        predictions_report+="FOLD "+str(fold) +" RESULTS\n\n"
        #GET THE FOLD TRAIN AND TEST INDEXES
        test_users_indexes=indexes[first_index:fold*users_per_fold]
        train_users_indexes=indexes[0:first_index]+indexes[fold*users_per_fold:len(indexes)]
        first_index=fold*users_per_fold

        #TRAIN PLDA WITH ALL THE TRAINING SUBSET POSTS
        train_users_df=concat_csv_reader(train_users_indexes,users_json,time_dir)
        train_df,preprocessed=process_data(train_users_df)
        labels = list(train_df["label"])
        vocab, vocab_n = vocab_size(preprocessed)
        alpha = 50 / int(num_topics) * (len(set(labels)))
        eta = 200 / vocab_n
        plda_model = createPLDA(0, 0, num_topics, alpha, eta, preprocessed, labels)
        plda_model.train(500)

        #GENERATE THE TOPIC LABEL DICT
        infoTopics = ""
        j = 0
        topic_label_dict = dict()
        for i in range(len(plda_model.topic_label_dict)):
            l = 0
            while (l < plda_model.topics_per_label and j < plda_model.k):
                infoTopics = infoTopics + plda_model.topic_label_dict[i] + ": "
                tuplas = plda_model.get_topic_words(j, int(n))
                topic_sentiment_score = (sentiment_score(tuplas))
                for k in range(len(tuplas)):
                    tupla = tuplas[k]
                    palabra = tupla[0]
                    probabilidad = str(tupla[1])
                    infoTopics = infoTopics + palabra + "," + probabilidad + "\t"
                infoTopics = infoTopics + "\n"
                topic_label_dict[j] = {"label": plda_model.topic_label_dict[i],
                                       "sentiment_score": topic_sentiment_score}
                j += 1
                l += 1

        #GENERATE THE TRAINING VECTORS
        train_vectors=[]
        train_labels=[]
        for train_index in train_users_indexes:
            user = list(users_json.keys())[train_index]
            df = user_csv_reader(user, users_json, time_dir)
            df, preprocessed_docs = process_data(df)
            lista_topics = []
            for i in range(len(preprocessed_docs)):
                topics = obtenerVectorTopics(plda_model, preprocessed_docs[i])
                lista_topics.append(list(topics))
            train_vectors.append(get_post_rankings(lista_topics, topic_label_dict,False))
            train_labels.append(users_json[user]["risk_level"])

        #PREDICT TEST SET LABELS
        test_vectors=[]
        test_labels=[]
        for test_index in test_users_indexes:
            user=list(users_json.keys())[test_index]

            predictions_report+="USER: "+user+" predictions \n"
            df = user_csv_reader(user,users_json, time_dir)
            df, preprocessed_docs = process_data(df)
            lista_topics = []
            for i in range(len(preprocessed_docs)):
                topics = obtenerVectorTopics(plda_model, preprocessed_docs[i])
                lista_topics.append(list(topics))
            #SAVE PREDICTED TEST TOPICS
            df_topics = pd.DataFrame(
                data={"postid": df["postid"], "date": df["date"], "label": df["label"], "topics": lista_topics})
            filepath = Path(dest_dir + '/FOLD'+str(fold)+"/"+ user + "_topics.tsv")
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df_topics.to_csv(filepath, sep='\t')
            test_vectors.append(get_post_rankings(lista_topics, topic_label_dict,False))
            test_labels.append(users_json[user]["risk_level"])

        #TRAIN CLASSIFIER WITH RANDOM FOREST
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        train_test_vectors=train_vectors+test_vectors
        longest_vector = max([len(elem) for elem in train_test_vectors])
        train_vectors=fill_with_zeros(train_vectors,longest_vector)

        clf.fit(np.array(train_vectors), train_labels)

        test_vectors=fill_with_zeros(test_vectors,longest_vector)
        predictions = clf.predict(test_vectors)

        report=classification_report(test_labels, predictions)
        report_dict = classification_report(test_labels, predictions,output_dict=True)
        predictions_report+="Real labels: "+str(test_labels)+"\n"
        predictions_report+="Predicted labels: "+str(predictions)+"\n"

        predictions_report+=report+"\n"
        all_macro.append(report_dict["macro avg"])
        all_weighted.append(report_dict["weighted avg"])


    macro_precision=0
    macro_recall=0
    macro_fscore=0
    weighted_precision=0
    weighted_recall=0
    weighted_fscore=0
    for m in range(len(all_macro)):
        macro_precision+=all_macro[0]["precision"]
        macro_recall+=all_macro[0]["recall"]
        macro_fscore+=all_macro[0]["f1-score"]
        weighted_precision += all_weighted[0]["precision"]
        weighted_recall += all_weighted[0]["recall"]
        weighted_fscore += all_weighted[0]["f1-score"]

    predictions_report+="OVERALL METRICS"+"\n"
    predictions_report+="macro precision: "+str(macro_precision/len(all_macro))+" macro recall: "+str(macro_recall/len(all_macro))+\
                        " macro f1-score: "+str(macro_fscore/len(all_macro))+"\n"+"weighted precision: "+\
                        str(weighted_precision/len(all_weighted))+" weighted recall: "+\
                        str(weighted_recall/len(all_weighted))+" weighted f1-score: "+\
                        str(weighted_fscore/len(all_weighted))



    with open(dest_dir + '/report.txt', 'w') as outfile:
        outfile.write(predictions_report)

