import os
import argparse
import time
import json
import numpy as np

from tqdm import tqdm
from collections import Counter, defaultdict
from retriever import Retriever
from WikiData import MyWikiData

class Data(object):

    def __init__(self,  args, wikidata):
        self.data_path = "data/{}/{}-{}.qa.json".format(args.data, args.data, args.data_type)
        self.tagme_path = "data/{}/{}-{}.tagme.json".format(args.data, args.data, args.data_type)

        self.wikidata = wikidata
        self.retriever = Retriever(args)

        with open(self.data_path, 'r') as f:
            orig_data = json.load(f)

        with open(self.tagme_path, 'r') as f:
            tagme_data = json.load(f)

        assert len(orig_data)==len(tagme_data)
        print ("Loaded {} QA data".format(len(orig_data)))

        self.save_path = "data/{}/{}-{}.retrieved.json".format(args.data, args.data, args.data_type)

        #### data to save ###
        n_cross_relations = []
        n_inner_relations = []
        n_total_relations = []
        data_to_save = []

        N_TFIDF = 5 if args.data=="webquestions" else 10
        N_BM25 = 40 if args.data=="webquestions" else 80

        for i, (d, tags) in tqdm(enumerate(zip(orig_data, tagme_data))):
            if len(tags)>0:
                sorted_tags = sorted(tags, key=lambda x: -x['score']) if 'score' in tags[0] else tags.copy()
                tags = []
                for e in sorted_tags:
                    # for some reason, tagme keeps tagging "The Who" for "who" questions.
                    # we will exclude them.
                    if not ((e['entity']=='The Who' and e['mention']=='who') or e["entity"]=="song"):
                        if e['entity'] not in tags:
                            tags.append(e['entity'])

            tfidf_docs = self.retriever.get_titles_from_query(d['question'], N_TFIDF)
            for t in tfidf_docs:
                if t not in tags:
                    tags.append(t)
            keywords = self.wikidata.populate(tags, k=args.n_hops, use_aliases=False)
            collected_docs = set()
            collected_paragraphs = []
            paragraphs_to_run_bm25 = []
            for (doc_name, hop, relation) in keywords[:80]:
                if doc_name in collected_docs:
                    continue
                collected_docs.add(doc_name)
                contents = self.retriever.get_contents_from_title(doc_name,
                                                                n_words=self.retriever.get_n_words(d['question'], doc_name),
                                                                only_first=hop>0)
                if len(contents)==0:
                    continue
                collected_paragraphs.append((contents[0], hop, relation))
                assert hop==0 or len(contents)==1
                paragraphs_to_run_bm25 += [(content, relation) for content in contents[1:]]

            collected_paragraphs = [par for i, par in sorted(enumerate(collected_paragraphs),
                                                        key=lambda x: (x[1][1], x[1][0][1], x[0]))]
            bm25_paragraphs = self.retriever.get_paragraphs_from_documents(d['question'],
                                                                    paragraphs_to_run_bm25,
                                                                    N_BM25,
                                                                    only_first=False,
                                                                    is_tuple=True)
            pars = [(par, rel) for par, hop, rel in collected_paragraphs if hop==0]
            pars_1 = [(par, rel) for par, hop, rel in collected_paragraphs if hop==1]
            for p_i in range(len(bm25_paragraphs)):
                if len(pars_1)>p_i:
                    pars.append(pars_1[p_i])
                pars.append(bm25_paragraphs[p_i])
            pars += self.retriever.get_paragraphs_from_documents(d['question'],
                                                                 pars_1[len(bm25_paragraphs):],
                                                                 100,
                                                                 only_first=False, is_tuple=True)
            pars += self.retriever.get_paragraphs_from_documents(d['question'],
                                                                 [(par, rel) for par, hop, rel in collected_paragraphs if hop>1],
                                                                 100,
                                                                 only_first=False, is_tuple=True)
            # truncate pars to be 100 at maximum
            pars = pars[:100]

            relations = [p[1] for p in pars]
            pars = [p[0] for p in pars]

            # get graph information for the GrpahReader
            collected_docs = set([par[0] for par in pars])
            graph = self.wikidata.get_graph(collected_docs)
            constructed_graph = {}
            n_cross, n_inner = 0, 0
            for i1, (title1, index1, _) in enumerate(pars):
                for i2, (title2, index2, _) in enumerate(pars):
                    if i1==i2: continue
                    if (title1, title2) in graph and index1==index2==0:
                        constructed_graph[(i1, i2)] = graph[(title1, title2)]
                        n_cross += 1
                    if title1==title2 and index1==0 and index2>0:
                        constructed_graph[(i1, i2)] = ["<CHILD_PARAGRAPH>"]
                        constructed_graph[(i2, i1)] = ["<PARENT_PARAGRAPH>"]
                        n_inner += 2
            n_cross_relations.append(n_cross)
            n_inner_relations.append(n_inner)
            n_total_relations.append(n_cross+n_inner)
            data_to_save.append(json.dumps({
                    'question': d['question'],
                    'answers': d['answers'],
                    'paragraphs': pars,
                    'graph': {'{} {}'.format(k[0], k[1]): v for k, v in constructed_graph.items()}
                }))

        print ("Cross", np.mean(n_cross_relations))
        print ("Inner", np.mean(n_inner_relations))
        print ("Total", np.mean(n_total_relations))

        with open(self.save_path,  'w') as f:
            f.write("\n".join(data_to_save))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="webquestions")
    parser.add_argument('--data_type', type=str, default="dev")
    parser.add_argument('--tfidf_path', type=str,
                        default="/data/sewon/wikipedia/docs-tfidf.npz")
    parser.add_argument('--dump_path', type=str, default="/data/sewon/wikidata-20190708-all.json.bz2")
    parser.add_argument('--data_dir', type=str, default="/data/sewon/MyWikidata")
    parser.add_argument('--n_hops', type=int, default=2)
    parser.add_argument('--new', action="store_true")
    parser.add_argument('--wiki_db_path', type=str,
                        default="/data/sewon/wikipedia/docs.db")
    #parser.add_argument('--vocab_file', type=str,
    #                    default="/data/home/sewon/bert_vocab.txt")
    args = parser.parse_args()
    wikidata = MyWikiData(args)
    data = Data(args, wikidata)


