import os
import csv
import codecs
import logging
import argparse
from cornell_processing_utils import loadConversations, loadLines, extractSentencePairs

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