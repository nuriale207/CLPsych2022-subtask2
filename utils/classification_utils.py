import random

import numpy as np
from itertools import repeat

from utils.topics_metrics import jensenShannonDistance

def rule_based_classification(user_topics, topic_label_dict,convert_from_string=True):
    """
    Given a users topics it predicts a label for each topic based on the labels related with the most salient topics and
    the distance with the previous topic
    :param user_topics: a list containing a users' topics
    :param topic_label_dict: a dictionary with a one to one correspondence between the topic and a label
    :return: a list with the inferred labels for each timeline post
    """
    #First we decide that the first label will be the one related with the most salient topic of the first post topic
    # distribution
    labels=[]
    if convert_from_string:
        users_topics=[]
        for user_topic in user_topics:
            user_topic=user_topic.replace('[','')
            user_topic=user_topic.replace(']','')
            user_topic=user_topic.split(",")
            user_topic=[float(i) for i in user_topic]
            users_topics.append(user_topic)
    else:
        users_topics=user_topics

    first_label=topic_label_dict[str(users_topics[0].index(max(users_topics[0])))]
    labels.append(first_label)
    for first_index in range(len(users_topics)-1):
        second_index=first_index+1
        v1=users_topics[first_index]
        v2=users_topics[second_index]
        distance=jensenShannonDistance(v1,v2)

        related_label=topic_label_dict[str(users_topics[second_index].index(max(users_topics[second_index])))]
        if(distance<=0.5 and related_label==first_label and (related_label=="0" or related_label=="IE")):
            labels.append(first_label)
        elif(distance<=0.5 and related_label==first_label and (related_label=="IS")):
            labels.append("0")
            first_label="0"
        elif(distance>0.5 and (related_label=="IE")):
            labels.append("IE")
            first_label="IE"

        elif(distance>0.5 and (related_label=="IS")):
            labels.append("IS")
            first_label="IS"
        else:
            labels.append("0")
            first_label="0"


    return labels

def arg_max_classification(user_topics, topic_label_dict,convert_from_string=True):
    """
    Given a users topics it predicts a label for each topic based on the labels related with the most salient topics and
    the distance with the previous topic
    :param user_topics: a list containing a users' topics
    :param topic_label_dict: a dictionary with a one to one correspondence between the topic and a label
    :return: a list with the inferred labels for each timeline post
    """
    #First we decide that the first label will be the one related with the most salient topic of the first post topic
    # distribution
    labels=[]
    if convert_from_string:
        users_topics=[]
        for user_topic in user_topics:
            user_topic=user_topic.replace('[','')
            user_topic=user_topic.replace(']','')
            user_topic=user_topic.split(",")
            user_topic=[float(i) for i in user_topic]
            users_topics.append(user_topic)
    else:
        users_topics=user_topics

    for id in range(len(users_topics)):
        labels.append(topic_label_dict[str(users_topics[id].index(max(users_topics[id])))])

    return labels



def get_random_indexes(num_users):
    return random.sample(range(0,num_users),num_users)


def get_post_rankings(user_topics, topic_label_dict, convert_from_string=True):
    """
       Given a users topics at post level it predicts a ranking of its' posts in therms of mood level.
       :param user_topics: a list containing a users' topics
       :param topic_label_dict: a dictionary with a one to one correspondence between the topic and a label
       :return: a list with the inferred labels for each timeline post
       """
    labels = []
    users_topics = []
    if convert_from_string:

        for user_topic in user_topics:
            user_topic = user_topic.replace('[', '')
            user_topic = user_topic.replace(']', '')
            user_topic = user_topic.split(",")
            user_topic = [float(i) for i in user_topic]
            users_topics.append(user_topic)
    else:
        users_topics = user_topics

    user_scores=[]
    for topic in users_topics:
        score=0
        for id in range(len(topic)):
            topic_w=topic[id]
            if (convert_from_string):
                topic_score=topic_label_dict[str(id)]["sentiment_score"]
            else:
                topic_score=topic_label_dict[id]["sentiment_score"]

            score+=topic_w*topic_score
        user_scores.append(score)
    return user_scores

def fill_with_zeros(topics_list,length):
    for topics in topics_list:
        topics.extend(list(repeat(float(0),length-len(topics))))
    return topics_list