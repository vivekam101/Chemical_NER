from __future__ import unicode_literals, print_function

import spacy
from spacy.matcher import PhraseMatcher
import plac
from pathlib import Path
from spacy.tokens import Doc
import random
import re
import json
import sys
import pickle
from spacy.gold import GoldParse,biluo_tags_from_offsets
from spacy.scorer import Scorer
from spacy.util import decaying
from spacy.util import minibatch, compounding

nlp = spacy.load('en') # loading default spacy model for english
to_train_ents = []     # annotated dataset


def preprocess(line):
    """ preprocess data from wiki """

#    line = [y.lower_ for y
#            in line
#            if y.pos_ != 'PUNCT' and len(y)>2]
    
    line = ' '.join(str(v.lower_) for v in line)
#    line = line.replace('\n', ' ').replace('\r', '').replace('.', '')
#    line = re.sub(r'[\d+]+','',line)
#    line = re.sub(r"[^a-zA-Z0-9]+[0-9]+", ' ', line)
#    line = re.sub(r'[^\w]', ' ', line)
#    line = re.sub("\s\s+" , " ", line)
    return line

def offseter(lbl, doc, matchitem):
    """ generate offset for training set using spacy phrasematcher """

#    o_one = len(str(doc[0:matchitem[1]])) # since index starts from 0
    subdoc = doc[matchitem[1]:matchitem[2]]
#    o_two = o_one + len(str(subdoc))
    return (subdoc.start_char, subdoc.end_char, lbl)

def convertspacyapiformattocliformat(nlp, TRAIN_DATA):
    docnum = 1
    documents = []
    for t in TRAIN_DATA:
        doc = nlp(t[0])
        tags = biluo_tags_from_offsets(doc, t[1]['entities'])
        ner_info = list(zip(doc, tags))
        print(ner_info)
        tokens = []
        sentences = []
        for n, i in enumerate(ner_info):
            token = {"head" : 0,
            "dep" : "",
            "tag" : "",
            "orth" : i[0].string,
            "ner" : i[1],
            "id" : n}
            tokens.append(token)
        sentences.append({'tokens' : tokens})
        document = {}
        document['id'] = docnum
        docnum+=1
        document['paragraphs'] = []
        paragraph = {'raw': doc.text,"sentences" : sentences}
        document['paragraphs'] = [paragraph]
        documents.append(document)
    return documents

def load():
    """ prepare training set: preprocess, annotate, dataset(to_train_ents) preparation  """
    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    with open("chemical_names.pkl", "rb") as f:
        chemical_names = pickle.load(f)
        labels_dict = {'CHEM':chemical_names}

    count_dict ={}
    for i in labels_dict.keys():
        count_dict[i]=0
    for label in labels_dict.keys():
        ner.add_label(label)

    matcher = PhraseMatcher(nlp.vocab)
    for key,values in labels_dict.items():
        patterns = [nlp(text.lower()) for text in values]
        matcher.add(key, None, *patterns)

    res = []
    with open("chemicals_test.txt") as f:   # read text data for training
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            doc = nlp(chunk)
            for mnlp_line in doc.sents:
                mnlp_line = preprocess(mnlp_line)
                mnlp_line = nlp(mnlp_line)
                matches = matcher(mnlp_line)
    #        print(nlp.vocab.strings[378])
                res = [offseter(nlp.vocab.strings[x[0]], mnlp_line, x)
                        for x
                        in matches]

                for x in matches:
                    count_dict[nlp.vocab.strings[x[0]]]+=1

    #        for match_id, start, end in matches:
    #            rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
    #            res = [offseter(rule_id, mnlp_line, x)
    #                   for x
    #                   in matches]
                if res:
                    to_train_ents.append((str(mnlp_line), dict(entities=res)))
    print(to_train_ents)
    print(count_dict)
    print("training set length",len(to_train_ents))
    cli_set = (convertspacyapiformattocliformat(nlp,to_train_ents))
    with open('cli_set.json', 'w') as outfile:
        json.dump(cli_set, outfile)

if __name__ == '__main__':
    plac.call(load)
