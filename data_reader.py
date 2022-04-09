"""Read txt/csv file as input."""
import json

import pandas as pd
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
from gensim.parsing.preprocessing import preprocess_documents, remove_stopword_tokens, stem
import math

def json_reader(path):
    """
    Given a path with a json returns the json object with the information of the json
    :param path: path to the json
    :return: json object
    """
    # Opening JSON file
    f = open(path)

    # returns JSON object as
    # a dictionary
    fileJson = json.load(f)

    return fileJson

def csv_reader(path: str) -> pd.DataFrame:
    """Read & process TSV file.
    
    Parameters
    ----------
    path: str
        path to TSV file.

    Returns
    -------
    df: pd.DataFrame
        dataframe containing timelines of a user.
    """
    df = pd.read_csv(path,sep='\t')
    #df = process_data(df)

    return df

def all_csv_reader(users_json,timeline_dir):
    """
    Returns a pandas df object with the merged information of all the users timelines
    :param users_json: json with the information of the users
    :param timeline_dir: the directory containing the timelines tsv files
    :return: a pandas df object with the merged information of all the users timelines
    """
    i=0
    for user in users_json:

        if(i==0):
            user_df=user_csv_reader(user,users_json,timeline_dir)
            i+=1
        else:
            user_df=pd.concat([user_df,user_csv_reader(user,users_json,timeline_dir)],ignore_index=True)
    return user_df

def user_csv_reader(user_id,users_json,timeline_dir):
    i=0
    timelines = users_json[user_id]["timelines"]
    for timeline in timelines:
        if (i == 0):
            user_df = csv_reader(timeline_dir + "/" + timeline + ".tsv")
            i += 1
        else:
            user_df = pd.concat([user_df, csv_reader(timeline_dir + "/" + timeline + ".tsv")], ignore_index=True)
    return user_df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process input dataframe.

    Parameters
    ----------    
    df: pd.DataFrame
        Raw dataframe containing timelines.

    Returns
    -------
    df: pd.DataFrame
        processed dataframe
    """
    print(df.keys())
    documents=df["content"]
    titles=df["title"]
    df["label"]=df["label"].replace(0,"O")

    print(documents)
    print(titles)
    for i in range(len(documents)):
        if(isinstance(titles[i],float)==False):
            documents[i]=titles[i]+" "+documents[i]
    # df["content"]=titles
    preprocessed=preprocess_documents(list(df["content"]))
    for doc in range(len(preprocessed)):
        preprocessed[doc]=remove_stopword_tokens(preprocessed[doc])
        preprocessed[doc]=stem(" ".join(preprocessed[doc]))
        preprocessed[doc]=preprocessed[doc].split(" ")
    return df,preprocessed


