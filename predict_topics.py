import argparse
from pathlib import Path

import pandas as pd

from utils.data_reader import json_reader, process_data, user_csv_reader
from utils.generate_topics import obtenerVectorTopics, load_PLDA_model


def create_arguments():
    parser = argparse.ArgumentParser(
        description='This runnable preproccesses the documents and generates a json file for each user with the '
                    'predicted topics in each post. It also generates a json file with the most important words in each '
                    'topic and their weight.')

    parser.add_argument('--json', dest='json_file', nargs='+',
                        help='Path to the json file with the users information.')
    parser.add_argument('--timelines', dest='time_dir', nargs='+',
                        help='Path to the directory with the timelines tsv files.')
    parser.add_argument('--model', dest='model', nargs='+',
                        help='Path to the model.')


    parser.add_argument('--path',dest='path_dir', nargs='+',
                        help='Path to the directory to save the generated files.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_arguments()
    user_dir = args.json_file[0]
    time_dir = args.time_dir[0]
    model_path=args.model[0]

    dest_dir=args.path_dir[0]

    plda_model=load_PLDA_model(model_path)

    plda_model.train(500)
    users_json=json_reader(user_dir)


    all_users_topics=dict()
    for user in users_json:
        df=user_csv_reader(user,users_json,time_dir)
        df,preprocessed_docs=process_data(df)
        lista_topics=[]
        for i in range(len(preprocessed_docs)):
            topics=obtenerVectorTopics(plda_model,preprocessed_docs[i])
            lista_topics.append(list(topics))
        df_topics=pd.DataFrame(data={"postid":df["postid"],"date":df["date"],"label":df["label"],"topics":lista_topics})
        filepath = Path(dest_dir+'/'+user+"_topics.tsv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df_topics.to_csv(filepath,sep='\t')

    plda_model.save(dest_dir+'/plda_model.pkl')




