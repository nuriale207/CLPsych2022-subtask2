import random

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

def get_random_indexes(num_users):
    return random.sample(range(0,num_users),num_users)