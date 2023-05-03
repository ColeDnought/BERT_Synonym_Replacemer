#!/usr/bin/env python
import sys
import string
import nltk

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import tensorflow

import gensim
import transformers

from typing import List


def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()


def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    out = set()
    for syn in wn.synsets(lemma, pos=pos):
        for lem in syn.lemmas():
            if lem.name() != lemma:
                out.add((lem.name().replace("_", "-")).lower())
    return out


def wn_frequency_predictor(context: Context) -> str:
    freq = dict()
    for syn in wn.synsets(context.lemma, context.pos):
        for lex in syn.lemmas():
            if lex.name() in freq:
                freq[lex.name().replace("_", " ")] += lex.count()
            elif lex.name() != context.lemma:
                freq[lex.name().replace("_", " ")] = lex.count()
    return max(freq, key=freq.get)


def wn_simple_lesk_predictor(context: Context) -> str:
    overlap_dict = dict()
    for syn in wn.synsets(context.lemma, context.pos):
        words = tokenize(syn.definition())

        # add hypernyms
        for hyper in syn.hypernyms():
            words.append(hyper.name())
            words += tokenize(hyper.definition())

        # add examples
        for example in syn.examples():
            words += tokenize(example)

        # compute overlap
        for word in context.left_context:
            if word in words and word not in stopwords.words('english'):
                overlap_dict[syn.name()] = overlap_dict.get(syn.name(), 0) + 1
        for word in context.right_context:
            if word in words and word not in stopwords.words('english'):
                overlap_dict[syn.name()] = overlap_dict.get(syn.name(), 0) + 1

    if overlap_dict:
        return max(overlap_dict, key=overlap_dict.get).split('.')[0]
    else:
        return wn_frequency_predictor(context)


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self, context: Context) -> str:
        max_sim = 0
        most_sim = None
        words = get_candidates(context.lemma, context.pos)
        for word in words:
            if self.model.has_index_for(word):
                sim = self.model.similarity(context.lemma, word)
                if sim > max_sim:
                    max_sim = sim
                    most_sim = word
        return most_sim


class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.classifier = transformers.pipeline('sentiment-analysis')

    def predict(self, context: Context) -> str:
        uncoded = context.left_context + ['[MASK]'] + context.right_context
        input_toks = self.tokenizer.encode(uncoded)
        input_mat = np.array(input_toks).reshape((1, -1))

        predictions = self.model.predict(input_mat, verbose=False)[0]

        best_toks = np.argsort(predictions[0][len(context.left_context) + 1])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_toks)

        # Check word is in candidates
        for word in best_words:
            if word in get_candidates(context.lemma, context.pos):
                return word
        return None

    def sentiment_predict(self, context: Context) -> str:
        # Convert context into original sentence for sentiment analysis
        cont = context.left_context + [context.lemma] + context.right_context
        s = ''
        for w in cont:
            if w in string.punctuation:
                s += w
            else:
                s += ' '+w

        # Get sentiment (positive or negative, and confidence score)
        tone = self.classifier(s)

        uncoded = context.left_context + ['[MASK]'] + context.right_context + [tone[0]['label'], tone[0]['score']]
        input_toks = self.tokenizer.encode(uncoded)
        input_mat = np.array(input_toks).reshape((1, -1))

        predictions = self.model.predict(input_mat, verbose=False)[0]

        best_toks = np.argsort(predictions[0][len(context.left_context) + 1])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_toks)

        for word in best_words:
            if word in get_candidates(context.lemma, context.pos):
                return word
        return None


if __name__ == "__main__":

    ernie = BertPredictor()

    for context in read_lexsub_xml((sys.argv[1])):
        if len(sys.argv) > 2:
            if sys.argv[3] == 3: # Sentiment analysis addition
                prediction = ernie.sentiment_predict(context)
            if sys.argv[2] == 2: # Word2Vec Embeddings
                W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
                predictor = Word2VecSubst(W2VMODEL_FILENAME)
            if sys.argv[2] == 1: # Simple Lesk algorithm
                predictor = wn_simple_lesk_predictor(context)
        else: # BERT masked language model
            prediction = ernie.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))