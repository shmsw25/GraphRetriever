import json
import tagme
import argparse

from tqdm import tqdm

class TAGME(object):

    def __init__(self, gcube_token):
        tagme.GCUBE_TOKEN = gcube_token

    def extract_entities(self, question, threshold=0.1):
        if type(question)==list:
            return [self.extract_entities(_question) for _question in question]
        annotations = tagme.annotate(question)
        return [{'entity': ann.entity_title,
                 'entity_id': ann.entity_id,
                 'mention': ann.mention,
                 'score': ann.score}
                    for ann in sorted(annotations.get_annotations(threshold), key=lambda x: -x.score)]

    def extract_mentions(self, question):
        if type(question)==list:
            return [self.extract_mentions(_question) for _question in question]

        mentions = tagme.mentions(question)
        return [{'mention': mention.mention, 'score': mention.linkprob} \
                for mention in sorted(mentions.mentions, key=lambda x: -x.linkprob)]

    def get_semantic_relations(self, entity_pairs, is_id=False):
        if is_id:
            rels = tagme.relatedness_wid(entity_pairs)
        else:
            rels = tagme.relatedness_title(entity_pairs)
        return [{'entity1': rel.title1, 'entity2': rel.title2, 'score': rel.rel} \
                 for rel in rels.relatedness]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="webquestions")
    parser.add_argument('--data_type', type=str, default="dev")
    parser.add_argument('--gcube_token', type=str, default=None)
    args = parser.parse_args()

    mytagme = TAGME(args.gcube_token)
    with open('data/{}/{}-{}.qa.json'.format(args.data, args.data, args.data_type), 'r') as f:
        orig_data = json.load(f)
    data = []
    for d in tqdm(orig_data):
        data.append(mytagme.extract_entities(d['question']))
    with open('data/{}/{}-{}.tagme.json'.format(args.data, args.data, args.data_type), 'w') as  f:
        json.dump(data, f)
