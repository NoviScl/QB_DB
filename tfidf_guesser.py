from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import argparse
from os import path

from typing import Union, Dict
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from qanta_util.qbdata import QantaDatabase

kBIAS = "BIAS_CONSTANT"

MODEL_PATH = 'tfidf.pickle'
INDEX_PATH = 'index.pickle'
ANSWERS_PATH = 'answers.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


class TfidfGuesser:
    """
    Class that, given a query, finds the most similar question to it.
    """
    def __init__(self):
        """
        Initializes data structures that will be useful later.
        """        
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=True, analyzer="word", stop_words='english', max_df=0.5)
        self.tfidf_matrix = None
        self.tfidf = None
        self.answers = None

    def train(self, training_data: QantaDatabase, limit=-1) -> None:
        """
        Use a tf-idf vectorizer to analyze a training dataset and to process
        future examples.
        
        Keyword arguments:
        training_data -- The dataset to build representation from
        limit -- How many training data to use (default -1 uses all data)
        """
        
        questions = [x.text for x in training_data.guess_train_questions]
        answers = [x.page for x in training_data.guess_train_questions]
        self.answers = answers

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        self.tfidf_vectorizer = self.tfidf_vectorizer.fit(questions)
        self.tfidf = self.tfidf_vectorizer.transform(questions)

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump(self.tfidf, f)
        
        with open(ANSWERS_PATH, 'wb') as f:
            pickle.dump(self.answers, f)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        """
        Given the text of questions, generate guesses (a list of both both the page id and score) for each one.

        Keyword arguments:
        questions -- Raw text of questions in a list
        max_n_guesses -- How many top guesses to return
        """
        guesses = []
        if not max_n_guesses:
            max_n_guesses = 1
        
        question_tfidf = self.tfidf_vectorizer.transform(questions)
        cosine_similarities = cosine_similarity(question_tfidf, self.tfidf)
        for cos in cosine_similarities:
            indices = cos.argsort()[::-1]
            this_guess = []
            for i in range(max_n_guesses):
                idx = indices[i]
                this_guess.append((self.answers[idx], cos[indices[i]]))
            guesses.append(this_guess)

        return guesses

    def load(self):
        with open(MODEL_PATH, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        with open(INDEX_PATH, 'rb') as f:
            self.tfidf = pickle.load(f)
        
        with open(ANSWERS_PATH, 'rb') as f:
            self.answers = pickle.load(f)
        

        
        




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--guesstrain", default="/home/sichenglei/nlp-hw-master/data/small.guesstrain.json", type=str)
    parser.add_argument("--limit", default=-1, type=int)

    flags = parser.parse_args()

    print("Loading %s" % flags.guesstrain)
    guesstrain = QantaDatabase(flags.guesstrain)
    # guessdev = QantaDatabase(flags.guessdev)

    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(guesstrain, limit=flags.limit)    

    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.load()
    
    test_questions = ["Name this painter who painted Mona Lisa"]
    print (tfidf_guesser.guess(questions = test_questions, max_n_guesses = 1)[0][0][0])

   
    
