from data_reader import csv_reader
from generate_topics import createPLDA
from preprocess_data import preprocesado

if __name__ == "__main__":
    all_preprocessed=[]
    all_tags=[]
    vocab=set()
    for doc in ["57b1ec364b.tsv","b6e248adc6.tsv","b72eec29d1.tsv"]:
        dataset, preprocessed = csv_reader("training_dataset/train/"+doc)

        all_preprocessed.extend(preprocessed)
        all_tags.extend(list(dataset["label"]))
        for text in preprocessed:
            for word in text:
                vocab.add(word)
    #preprocessed=preprocesado(preprocessed,True,True,True)
    # print(preprocessed)
    # print(dataset)
    # print(dataset["title"][0])
    # print(dataset["content"][0])
    print(len(vocab))

    plda_model=createPLDA(0,0,2,12.5,0.36,all_preprocessed,all_tags)

    print("Topic words")
    print(plda_model.get_topic_words(0))
    print(plda_model.get_topic_words(1))
    print(plda_model.get_topic_words(2))
    print(plda_model.get_topic_words(3))
    print(plda_model.get_topic_words(4))
    print(plda_model.get_topic_words(5))

    print(plda_model.topic_label_dict)

    infoTopics = ""
    # print(str(len(modelo.topic_label_dict) * modelo.topics_per_label))
    j=0
    for i in range(len(plda_model.topic_label_dict)):
        l=0
        while(l<plda_model.topics_per_label and j<plda_model.k):
            infoTopics = infoTopics + plda_model.topic_label_dict[i] + ": "
            tuplas = plda_model.get_topic_words(j, 10)

            for k in range(len(tuplas)):
                tupla = tuplas[k]
                palabra = tupla[0]
                probabilidad = str(tupla[1])
                infoTopics = infoTopics + palabra + "," + probabilidad + "\t"
            infoTopics = infoTopics + "\n"
            j+=1
            l+=1


    print(infoTopics)