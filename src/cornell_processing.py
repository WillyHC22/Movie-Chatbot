import os
import csv
import codecs
import logging
import argparse

logger = logging.getLogger(__name__)

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

parser = argparse.ArgumentParser(description="Processing cornell movie data")
parser.add_argument("--conv_file", type=str, default="data/raw/cornell_movie/movie_conversations.txt",
                    help="path to the data file with movie conversations")
parser.add_argument("--lines_file", type=str, default="data/raw/cornell_movie/movie_lines.txt",
                    help="path to the data file with movie lines")
parser.add_argument("--save_file", type=str, default="data/processed/cornell_movie/processed_movie_lines.txt")
args = parser.parse_args()

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


if __name__ == "__main__":
    lines_file = args.lines_file
    conversations_file = args.conv_file
    save_file = args.save_file

    delimiter = "\t"
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    logger.info(f"Processing the lines from {lines_file}")
    lines = loadLines(lines_file, MOVIE_LINES_FIELDS)
    logger.info(f"Done processing lines. Now processing conversations from {conversations_file}")
    conversations = loadConversations(conversations_file, lines, MOVIE_CONVERSATIONS_FIELDS)

    logger.info(f"Done processing conversations. Now saving to a csv file in {save_file}")
    with open(save_file, "w", encoding="utf-8") as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator="\n")
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)