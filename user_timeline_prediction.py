import argparse
import json

from utils.classification_utils import get_post_rankings
from utils.data_reader import csv_reader


def create_arguments():
    parser = argparse.ArgumentParser(
        description='This runnable preproccesses the documents and generates a json file for each user with the '
                    'predicted topics in each post. It also generates a json file with the most important words in each '
                    'topic and their weight.')

    parser.add_argument('--topics', dest='topics', nargs='+',
                        help='Path to the df file that includes a users posts predicted topics.')
    parser.add_argument('--topic_label_dict', dest='label_dict', nargs='+',
                        help='Path to the directory with the json that has the correspondence between the topic index '
                             'and its related topic.')
    parser.add_argument('--path',dest='path_dir', nargs='+',
                        help='Path to the directory to save the generated predictions (they will be aded to de original '
                             'csv file).')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_arguments()
    topics_df = args.topics[0]
    topic_label_dict = args.label_dict[0]
    dest_dir = args.path_dir[0]

    topics = csv_reader(topics_df)
    f = open(topic_label_dict)
    topic_label_dict = json.load(f)
    get_post_rankings(topics["topics"], topic_label_dict)
    print(get_post_rankings(topics["topics"], topic_label_dict))
