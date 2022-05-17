import os
import re
import ast

def loadLines(fileName, fields):
    """
    Splits each line of fileName [str] into a dictionnary of fields, in our case with lineID-characterID-movieID-character-text
    Given that fields [list] is in the form metionned above (in order)
    returns:
    lines [dict]
    """
    lines = {}
    with open(fileName, "r", encoding='iso-8859-1') as datafile:
        for line in datafile:
            infos = line.split(" +++$+++ ")
            lineObj = {}

            for index, info in enumerate(infos):
                if index != 0:
                    lineObj[fields[index]] = info

            lines[infos[0]] = lineObj

    return lines


def loadConversations(fileName, lines, fields):
    """
    Take the lines form loadLines, and outputs the conversations based on movie_conversations.txt and fields character1ID-character2ID-movieID-utteranceIDs
    returns:
    conversations [list of list]
    """
    conversations = []
    with open(fileName, "r", encoding='iso-8859-1') as datafile:
        for line in datafile:
            convObj = {}
            infos = line.split(" +++$+++ ")

            for index, info in enumerate(infos):
                convObj[fields[index]] = info

            utteranceIDs = ast.literal_eval(infos[-1])
            convObj["lines"] = []
            for lineID in utteranceIDs:
                convObj["lines"].append(lines[lineID])

            conversations.append(convObj)

    return conversations


def extractSentencePairs(conversations):
    """
    take conversations defined previously and extract a list of pairs
    returns:
    qa_pairs [list]
    """
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"])-1):
            convLine1 = conversation["lines"][i]["text"].strip()
            convLine2 = conversation["lines"][i+1]["text"].strip()

            if convLine1 and convLine2: #In case one of the two is empty
                qa_pairs.append([convLine1, convLine2])

    return qa_pairs