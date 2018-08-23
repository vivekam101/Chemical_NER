from __future__ import unicode_literals, print_function

import spacy
from spacy.matcher import PhraseMatcher
import plac
from pathlib import Path
from spacy.tokens import Doc
import random
import re
import sys
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from spacy.util import decaying
from spacy.util import minibatch, compounding

nlp = spacy.load('en') # loading default spacy model for english
to_train_ents = []     # annotated dataset

def evaluate(ner_model, examples):
    """ return the score for ner_model against test set in examples"""

    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

def preprocess(line):
    """ preprocess data from wiki """
    
#    line = [y.lower_ for y
#            in line
#            if y.pos_ != 'PUNCT' and len(y)>2]
    
    line = ' '.join(str(v.lower_) for v in line)
    line = line.replace('\n', ' ').replace('\r', '').replace('.', '')
    line = re.sub(r'[\d+]+','',line)
    line = re.sub(r"[^a-zA-Z0-9]+[0-9]+", ' ', line)
    line = re.sub(r'[^\w]', ' ', line)
    line = re.sub("\s\s+" , " ", line)
    return line

def offseter(lbl, doc, matchitem):
    """ generate offset for training set using spacy phrasematcher """

    o_one = len(str(doc[0:matchitem[1]]))
    subdoc = doc[matchitem[1]:matchitem[2]]
    o_two = o_one + len(str(subdoc))
    return [o_one, o_two, lbl]

def load():
    """ prepare training set: preprocess, annotate, dataset(to_train_ents) preparation  """
    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    labels_dict = {'CHEMICAL': ['calcium sulfate', 'chlorine trifluoride', 'carbon dioxide', 'hydrogen sulfide', 'barium sulfate', 'iron sulfate', 'calcium chloride', 'potassium nitrate', 'dichlorine monoxide', 'sulfur trioxide', 'Zinc White', 'sodium bicarbonate', 'calcium oxide', 'aluminium oxide', 'sodium chloride', 'calcium nitrate', 'Alum', 'copper sulfate', 'sodium chlorate', 'Oxygen', 'Plaster', 'ammonium nitrate', 'Agalite', 'ammonium bicarbonate', 'Casein', 'ammonium carbonate', 'Hypochlorous Acid', 'Milk of Lime', 'Soda Ash', 'Calcium hydroxide', 'zinc oxide', 'pocket guide', 'zinc hydroxide', 'arsenic pentafluoride', 'iron oxide', 'metal oxide', 'vinyl chloride', 'sodium sulfate', 'Chlorine Dioxide', 'thionyl chloride', 'barium carbonate', 'Guar Gum', 'carbon monoxide', 'Titania', 'aluminium hydroxide', 'sodium alumina', 'Gypsum', 'zinc sulfide', 'magnesium hydroxide', 'Clay', 'Sodium Bisulfite', 'Talc', 'chlorine dioxide', 'Magnesium Bisulfite', 'ammonium chloride', 'agate', 'Caustic Soda', 'Sulfur', 'sodium alginate', 'magnesium sulfate', 'barium sulfide', 'Polyvinyl Alcohol', 'Sodium Peroxide', 'Alabaster', 'Barium Carbonate', 'hydrogen peroxide', 'calcium carbonate', 'Mineral White', 'dichlorine hexoxide', 'sodium algina', 'chlorine oxide', 'Zinc Hydrosulfite', 'sulfur dioxide', 'Zinc Sulfide', 'Chlorine', 'sodium aluminate', 'sulfur oxide', 'hydrogen fluoride', 'Sodium Aluminate', 'titanium dioxide', 'chlorine monofluoride', 'polyvinyl chloride', 'Chlorine Gas', 'hydrogen chloride', 'sodium peroxide', 'sodium carbonate', 'Silicone antifoam', 'calcium hydroxide', 'Annaline', 'calcium carbide', 'sodium hydroxide', 'carbon tetrachloride', 'Alganic Acid', 'Rosin Soap', 'metal chloride', 'aluminium sulfate', 'chlorine', 'florine']}

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
    with open("chemicals_.txt") as f:   # read text data for training
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            doc = nlp(chunk)
            for mnlp_line in doc.sents:
                mnlp_line = preprocess(mnlp_line)
                mnlp_line = nlp(mnlp_line)
            #print(mnlp_line)
                matches = matcher(mnlp_line)
    #        print(nlp.vocab.strings[378])
            #print(matches)
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
                    to_train_ents.append([str(mnlp_line), dict(entities=res)])
    print(to_train_ents)
    print(count_dict)
    print("training set length",len(to_train_ents))


@plac.annotations(
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path))
def train(new_model_name='persons', output_dir=None):

    optimizer = nlp.begin_training()
    
    other_pipes = [pipe
                    for pipe
                    in nlp.pipe_names
                    if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(5):
            batches = minibatch(to_train_ents, size=compounding(4., 32., 1.001))
            losses = {}
            # for text, annotations in to_train_ents:
            #     nlp.update([text], [annotations], sgd=optimizer, drop=0.40,
            #                losses=losses)
            random.shuffle(to_train_ents)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=next(decaying(0.35, 0.25, 1e-4)),
                           losses=losses)
            print(losses)

    if output_dir is None:
        output_dir = "./model1"


    noutput_dir = Path(output_dir)
    if not noutput_dir.exists():
        noutput_dir.mkdir()
    if output_dir is not None:
        nlp.meta['accuracy'] = {'ner': best_acc}
    nlp.meta['name'] = new_model_name
    
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(output_dir)

    random.shuffle(to_train_ents)

        # quick test the saved model
    test_text = 'Gina Haspel, President Donald Trump’s controversial pick to be the next CIA director, has officially been confirmed by the Senate in a 54-45 vote.'
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(preprocess(nlp2(test_text)))
    print("Entities in '%s'" % doc2)
    for ent in doc2.ents:
        print(ent.label_, ent.text)

@plac.annotations(
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path))

def test(new_model_name='persons', output_dir=None):
    output_dir = "./model1"
    test_text ='Aryl chlorides may be prepared by the Friedel-Crafts halogenation, using chlorine and a Lewis acid catalyst. The haloform reaction, using chlorine and sodium hydroxide, is also able to generate alkyl halides from methyl ketones, and related compounds. Chlorine adds to the multiple bonds on alkenes and alkynes as well, giving di- or tetra-chloro compounds. However, due to the expense and reactivity of chlorine, organochlorine compounds are more commonly produced by using hydrogen chloride, or with chlorinating agents such as phosphorus pentachloride (PCl5) or thionyl chloride (SOCl2).'
#    test_text = ''
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text)
    for line in doc2.sents:
        line = preprocess(line)
        doc = nlp2(line)
        print("Entities in '%s'" % line)
        for ent in doc.ents:
            print(ent.label_, ent.text)



if __name__ == '__main__':
    print("load model or train (1/0)")
    bit = int(input())
    if bit:
        #plac.call(load)
        plac.call(test)
    else:
        plac.call(load)
        plac.call(train)
