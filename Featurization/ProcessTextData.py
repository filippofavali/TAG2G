"""
    This script contains supporting function for the text data processing.
    It is used in several other scripts coupling word embeddings to the ability
    of featurize text to be used in further processing

    other source of code authors:


    Test in the script:
    key1:

"""

import os.path
import io
import string
from tqdm import tqdm
import numpy as np
from Config.FeaturesConfig import features_config


class TextProcessor:

    def __init__(self, word2vec_dir:str):

        self.word2vector = self.load_word2vec(word2vec_dir)

    def __call__(self, text_path, crop_length):

        print(f"Text working '{text_path}'")

        # need to be inferred from something --> motion should set the minimum length (but what if motion is not the minimum?)
        textfeatures = np.zeros([crop_length, 300 + 2])
        textfeatures[:, -1] = 1
        assert os.path.isfile(text_path), f"Provided '{text_path}' is not a file"
        sentence = self.load_tsv_unclipped(text_path)

        for wi, (start, end, raw_word) in enumerate(sentence):
            has_laughter = "#" in raw_word
            start_frame = int(start * 30)
            end_frame = int(end * 30)
            textfeatures[start_frame:end_frame, -1] = 0

            word = raw_word.translate(str.maketrans('', '', string.punctuation))
            word = word.strip()
            word = word.replace("  ", " ")

            if len(word) > 0:
                if word[0] == " ":
                    word = word[1:]

            if " " in word:
                ww = word.split(" ")
                subword_duration = (end_frame - start_frame) / len(ww)
                for j, w in enumerate(ww):
                    vector = self.word2vector.get(w)
                    if vector is not None:
                        ss = start_frame + int(subword_duration*j)
                        ee = start_frame + int(subword_duration*(j+1))
                        textfeatures[ss:ee, :300] = vector
            else:
                vector = self.word2vector.get(word)
                if vector is not None:
                    textfeatures[start_frame:end_frame, :300] = vector
            textfeatures[start_frame:end_frame, -2] = has_laughter

        return textfeatures

    def load_word2vec(self, fname):

        print(">> Text processor: loading word2vector, might require some minutes ...")
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in tqdm(fin, "loading crawl-300d-2M.vec"):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array([float(v) for v in tokens[1:]])
        return data

    def load_tsv_unclipped(self, tsvfile):
        sentence = []
        with open(tsvfile, "r") as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split("\t")
                if len(line) == 3:
                    start, end, raw_word = line
                    start = float(start)
                    end = float(end)
                    sentence.append([start, end, raw_word])
        return sentence


if __name__ == "__main__":

    # tested 11-12-23 working

    key1 = True
    if key1:
        parameters = features_config()
        test_processor = TextProcessor(parameters.word2vec_dir)
        random_crop = 180
        assert type(random_crop) == int, "Not an INT"
        text_path = r"C:\Users\faval\genea2023_dataset\trn\main-agent\tsv\trn_2023_v0_000_main-agent.tsv"
        text_features = test_processor(text_path, random_crop)
        print(text_features.shape)
        print(text_features)



