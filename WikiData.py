import os
import bz2
import json
import argparse
import numpy as np
import time
import re
import string

from tqdm import tqdm
from collections import defaultdict, Counter
from IPython import embed

from qwikidata.entity import WikidataItem, WikidataProperty
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.utils import dump_entities_to_json

from Database import MyDatabase
from joblib import Parallel, delayed
from multiprocessing import Pool
#from joblib import wrap_non_picklable_objects

SEP = "@@MYSEP@@"
PERIOD = 1000000

class MyWikiData(object):

    def __init__(self, args, load=False, connect_each=False):
        self.data_dir = args.data_dir
        self.db = MyDatabase(os.path.join(args.data_dir, 'wikidata.db'),
                             connect_each=connect_each)

        self.db.create(load, 'Entities', [('id', 'TEXT'),
                                    ('description', 'TEXT'), ('enwiki_link', 'TEXT'),
                                    ('texts', 'TEXT'), ('texts_norm', 'TEXT')])
        self.db.create(load, 'Properties', [('id', 'TEXT'),
                                    ('description', 'TEXT'), ('texts', 'TEXT')])
        self.db.create(load, 'Claims', [('id', 'TEXT'), ('property_id', 'TEXT'),
                                    ('value', 'TEXT'), ('valuetype', 'TEXT')])
        self.db.create(load, 'Text2Entity',  [('text', 'TEXT'), ('entity_id', 'TEXT')])
        self.db.create(load, 'Text2Entity_norm',  [('text', 'TEXT'), ('entity_id', 'TEXT')])

        self.none_property_ids = set()
        self.non_none_property_ids = set()

        if load:
            self.wjd = WikidataJsonDump(args.dump_path)

        def process(ii, entity_dict):
            if args.start>ii:
                return []

            if entity_dict["type"] == "item":
                return self.handle_entity(WikidataItem(entity_dict))
            elif entity_dict["type"] == "property":
                return self.handle_property(WikidataProperty(entity_dict))
            else:
                return []

        if load and args.n_processes==1:
            start_time=time.time()
            queries = defaultdict(list)
            for i, b in enumerate(self.wjd):
                for table_name, row in process(i, b):
                    #self.db.insert(table_name, row)
                    queries[table_name] += row
                if (i+1) % PERIOD == 0:
                    for table_name, rows  in queries.items():
                        self.db.insert(table_name, rows)
                    queries = defaultdict(list)
                    self.save_data(i+1)
                    if PERIOD>10000:
                        print ("%d mins"%((time.time()-start_time)/60))
                    else:
                        print ("%d secs"%(time.time()-start_time))
                    start_time=time.time()
            self.save_data(i+1)
        elif load:
            process = CloudpickleWrapper(process)
            outputs = []
            start_time = time.time()
            with Pool(args.n_processes) as pool:
                for i, b in enumerate(self.wjd):
                    outputs.append(pool.apply_async(process, (i, b)))
                    if (i+1) % PERIOD == 0:
                        queries = defaultdict(list)
                        for output in outputs:
                            for table_name, row in output.get():
                                queries[table_name]+=row
                        for table_name, rorws in queries.items():
                            self.db.insert(table_name, rows)
                        self.save_data(i+1)
                        if PERIOD>10000:
                            print ("%d mins"%((time.time()-start_time)/60))
                        else:
                            print ("%d secs"%(time.time()-start_time))
                        outputs = []
                        start_time=time.time()
            if len(outputs)>0:
                for output in outputs:
                    for table_name, row in output.get():
                        self.db.insert(table_name, row)
                self.save_data(i+1)

    def save_data(self, step):
        print ("=====\tSTEP = {}\t=====".format(step))
        print ("\t".join(self.db.commit()))

    def handle_entity(self, entity):
        entity_id = entity.entity_id
        enwiki_link = entity.get_sitelinks().get('enwiki', {}).get('title', None)
        all_claims = defaultdict(list)
        for property_id, claims in entity.get_truthy_claim_groups().items():
            for claim in claims:
                assert property_id==claim.mainsnak.property_id
                if not self.filter_mainsnak(claim.mainsnak):
                    all_claims[property_id].append(self.handle_mainsnak(claim.mainsnak))
        #assert entity_id not in self.entities
        texts = [entity.get_label()] + entity.get_aliases() + [enwiki_link]
        texts = list(set([t for t in texts if t is not None]))
        texts_norm = list(set([normalize_answer(t) for t in texts]))

        return [('Entities', [(entity_id, entity.get_description(), enwiki_link,
                               self.list2string(texts), self.list2string(texts_norm))]),
                ('Claims', [(entity_id, property_id,
                    self.value2string(statement['value'], statement['valuetype']),
                    statement['valuetype'])
                    for property_id, statements in all_claims.items()
                            for statement in statements])] + \
            [('Text2Entity', [(text, entity_id) for text in texts])] + \
            [('Text2Entity_norm', [(text, entity_id) for text in texts_norm])]

    def list2string(self, _list):
        return SEP.join(_list)

    def value2string(self, value, valuetype):
        if type(value)==str:
            return value
        if type(value)==dict and valuetype=='wikibase-entityid':
            return value['id']
        elif type(value)==dict and valuetype=='monolingualtext':
            return value['text']
        return json.dumps(value)

    def handle_property(self, property):
        #assert property.entity_id not in self.properties
        texts = [property.get_label()] + property.get_aliases()
        #self.db.insert('Properties', [(property.entity_id, property.get_description(), texts)])
        return [('Properties', [(property.entity_id, property.get_description(), self.list2string(texts))])]

    def filter_mainsnak(self, snak):
        if snak.snaktype!='value':
            return True
        return snak.snak_datatype in ['external-id', 'url', 'commonsMedia',
                                      'globe-coordinate', 'math', 'musical-notation',
                                      'time', 'geo-shape']

    def handle_mainsnak(self, snak):
        return {'datatype': snak.snak_datatype,
                'valuetype': snak.value_datatype,
                'value': snak.datavalue.value}

    def filter_entity_ids(self, entity_ids):
        filtered_entity_ids = []
        for e_id in entity_ids:
            e = self.get_entity(e_id)
            if e is not None and e['enwiki_link'] is not None and \
                    (not e['enwiki_link'].startswith('Category:')) and \
                    (not e['enwiki_link'].endswith(' (disambiguation)')) and \
                    (not e['enwiki_link'].endswith(' (word)')):
                filtered_entity_ids.append(e_id)
        return filtered_entity_ids

    def get_entities_from_text(self, text):
        def _get_entities_from_text(norm):
            rows = self.db.fetch('Text2Entity_norm' if norm else 'Text2Entity', 'text',
                                normalize_answer(text) if norm else text)
            if len(rows)==0:
                return set()
            entity_ids = [text for row in rows for text in row[1].split(SEP)]
            return self.filter_entity_ids(entity_ids)
        result = _get_entities_from_text(norm=False)
        if len(result)==0:
            result = _get_entities_from_text(norm=True)
        return result

    def get_entities_from_title(self, text):
        rows = self.db.fetch('Entities', 'enwiki_link', text)
        if len(rows)==0:
            return set()
        return set([r[0] for r in rows])

    def get_texts_from_entity(self, entity_id):
        entity = self.get_entity(entity_id)
        return set(entity['texts'])|set(entity['texts_norm'])

    def get_entity(self, entity_id):
        if type(entity_id)==list or type(entity_id)==set:
            return [self.get_entity(_entity_id) for _entity_id in entity_id]
        rows  = self.db.fetch('Entities', 'id', entity_id)
        assert len(rows)<=1
        if len(rows)==0:
            return None
        return {'description': rows[0][1], 'enwiki_link': rows[0][2],
                'texts': rows[0][3].split(SEP), 'texts_norm': rows[0][4].split(SEP)}

    def get_property(self, property_id):
        if type(property_id)==list:
            return  [self.get_property(i) for  i in property_id]
        rows = self.db.fetch('Properties', 'id', property_id)
        if len(rows)==0:
            return None
        assert len(rows)==1
        return {'description': rows[0][1], 'texts': rows[0][2].split(SEP)}

    def get_neighbors(self, entity_id):
        rows = self.db.fetch('Claims', 'id', entity_id)
        return [{'property_id': row[1], 'value': row[2], 'valuetype': row[3]} for row in rows]

    def populate(self, seed_texts, k=2, use_aliases=False):
        all_entities = []
        collected_titles = [(text, 0, 'seed') for text in seed_texts]
        new_entities = []
        for text in seed_texts:
            if use_aliases:
                for e in self.get_entities_from_text(text):
                    if e not in all_entities and e not in new_entities:
                        new_entities.append(e)
            else:
                for e in self.get_entities_from_title(text):
                    if e not in all_entities and e not in new_entities:
                        new_entities.append(e)
        for hop in range(k):
            added_entities = []
            for entity_id1 in new_entities:
                results = self.get_neighbors(entity_id1)
                for result in results:
                    if result['valuetype']=='wikibase-entityid' and \
                            result['value'].startswith('Q'):
                        entity_id2 = result['value']
                        if entity_id2 in all_entities or entity_id2 in new_entities \
                                or result['property_id'] in self.none_property_ids:
                            continue
                        if result['property_id'] not in self.non_none_property_ids:
                            if self.get_property(result['property_id']) is None:
                                self.none_property_ids.add(result['property_id'])
                                continue
                            self.non_none_property_ids.add(result['property_id'])
                        added_entities.append(entity_id2)
                        ent2 = self.get_entity(entity_id2)
                        if ent2 is not None and ent2['enwiki_link'] is not None and \
                                not any([ent2['enwiki_link']==a[0] for a in collected_titles]):
                            collected_titles.append((ent2['enwiki_link'], hop+1, result['property_id']))
                            if len(collected_titles)>=80:
                                break
            all_entities += new_entities
            new_entities = added_entities
            if len(collected_titles)>=80:
                break
        return collected_titles

    def get_graph(self, doc_names):
        graph = {}
        for i, doc_name in enumerate(doc_names):
            for entity in self.get_entities_from_title(doc_name):
                for result in self.get_neighbors(entity):
                    if result['valuetype']=='wikibase-entityid' and \
                            result['value'].startswith('Q'):
                        e = self.get_entity(result['value'])
                        if e is None:
                            continue
                        if self.get_property(result['property_id']) is None:
                            continue
                        title = e['enwiki_link']
                        if title in doc_names:
                            graph[(doc_name, title)] = self.get_property(result['property_id'])
        return graph

    def get_neighbors(self, entity_id):
        rows = self.db.fetch('Claims', 'id', entity_id)
        return [{'property_id': row[1], 'value': row[2], 'valuetype': row[3]} for row in rows]

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_path', type=str, default="/data/home/sewon/wikidata-20190708-all.json.bz2")
    parser.add_argument('--data_dir', type=str, default="/data/home/sewon/MyWikidata")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--n_processes', type=int, default=10)
    args = parser.parse_args()
    wikidata = MyWikiData(args, load=True)





