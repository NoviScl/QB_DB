from tfidf_guesser import *

class test:
    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
    
    def guess(self, question):
        return self.model.guess(questions=question, max_n_guesses=1)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--guesstrain", default="data/qanta.train.2018.04.18.json", type=str)
    # parser.add_argument("--guessdev", default="data/small.guessdev.json", type=str)
    # parser.add_argument("--limit", default=-1, type=int)

    # flags = parser.parse_args()

    # print("Loading %s" % flags.guesstrain)
    # guesstrain = QantaDatabase(flags.guesstrain)
    # # guessdev = QantaDatabase(flags.guessdev)
    
    # tfidf_guesser = TfidfGuesser()
    # tfidf_guesser.train(guesstrain, limit=flags.limit)
    
    # ## save checkpoint
    # with open(MODEL_PATH, 'wb') as f:
    #     pickle.dump(tfidf_guesser, f)

    # with open('tfidf.pickle', 'rb') as f:
    #     tfidf_guesser = pickle.load(f)
    
    test_questions = ["Name this painter who painted Mona Lisa"]
    # print (tfidf_guesser.guess(questions = test_questions, max_n_guesses = 1)[0][0][0])

    handler = test()
    handler.load('tfidf.pickle')
    print (handler.guess(test_questions))

