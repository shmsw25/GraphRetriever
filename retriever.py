import time
import math
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter

import drqa_retriever as retriever
from drqa_retriever import DocDB
from rank_bm25 import BM25Okapi
from Database import MyDatabase

from pytorch_transformers import BertTokenizer, BasicTokenizer

title_s = "<t>"
title_e = "</t>"

SEP1 = "<@@SEP@@>"
SEP2 = "<##SEP##>"
SEP3 = "<$$SEP$$>"


class Retriever(object):

    def __init__(self, args, need_vocab=True):
        self.tfidf_path=args.tfidf_path
        self.ranker = retriever.get_class('tfidf')(tfidf_path=self.tfidf_path)
        self.first_para_only = False
        self.db = DocDB(args.wiki_db_path)
        self.L = 300
        self.first_para_only = False

        if need_vocab:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            btokenizer = BasicTokenizer()
            self.tokenize = lambda c, t_c: tokenizer.tokenize(c)
            self.btokenize  = btokenizer.tokenize

        self.keyword2title = defaultdict(list)
        self.cache = {}

    def get_titles_from_query(self, query, n_docs):
        try:
            doc_names, doc_scores = self.ranker.closest_docs(query, n_docs)
        except Exception:
            return []
        return doc_names

    def get_contents_from_title(self, doc_name, n_words, only_first):
        if doc_name in self.cache:
            contents = self.cache[doc_name]
        else:
            try:
                contents = self.db.get_doc_text(doc_name).split('\n\n')
            except Exception:
                return []
            if contents[0]==doc_name:
                contents = contents[1:]
            contents = [c for c in contents if len(c.strip())>0]
            for i, c in enumerate(contents):
                t_c = self.btokenize(c)
                t_c2 = self.tokenize(c, t_c)
                contents[i] = "{}{}{}".format(SEP1.join(t_c), SEP2, SEP1.join(t_c2))
            contents = SEP3.join(contents)
            self.cache[doc_name] = contents
        if len(contents)==0:
            return []
        contents = [[ci.split(SEP1) for ci in c.split(SEP2)] for c in contents.split(SEP3)]
        return self.get_preprocessed_paragraphs(doc_name, contents.copy(), n_words=n_words,
                                                only_first=only_first)


    def get_preprocessed_paragraphs(self, doc_name, contents, n_words, only_first=False):
        curr_paragraphs = []
        curr_lengths =  []
        for tokenized_par, tokenized_par2 in contents:
            l = len(tokenized_par2)
            if len(curr_lengths)>0 and l<=n_words-curr_lengths[-1]-3:
                curr_paragraphs[-1] += ["<p>"]
                offset = l-len(tokenized_par)
                assert offset>=0
                curr_paragraphs[-1] += tokenized_par.copy()
                curr_lengths[-1] += l if curr_lengths[-1]==0 else l+3
            else:
                if l>n_words:
                    offset = n_words-len(tokenized_par2)+len(tokenized_par)
                    if offset<=n_words/2.0:
                        continue
                    tokenized_par = tokenized_par[:offset].copy()
                curr_paragraphs.append(tokenized_par.copy())
                curr_lengths.append(l)
            #assert curr_lengths[-1]<=n_words
            if only_first and len(curr_paragraphs)>1:
                curr_paragraphs = curr_paragraphs[:1]
                break
        tok_doc_name = self.btokenize(doc_name)
        return [[doc_name, i, [title_s] + tok_doc_name + [title_e] + t]
                for i, t in enumerate(curr_paragraphs)]

    def get_paragraphs_from_documents(self, query, _paragraphs, n_paragraphs,
                                      only_first=False, is_tuple=False):

        if len(_paragraphs)==0 or only_first:
            return _paragraphs

        if is_tuple:
            relations = [p[1] for p in _paragraphs]
            _paragraphs = [p[0] for p in _paragraphs]

        bm25 = BM25Okapi([p[2] for p in _paragraphs])
        paragraphs = []
        for index, score in sorted(enumerate(bm25.get_scores(self.btokenize(query)).tolist()),
                                key=lambda x: (-x[1], x[0])):
            if score==0 or len(paragraphs)==n_paragraphs:
                break
            if is_tuple:
                paragraphs.append((_paragraphs[index], relations[index]))
            else:
                paragraphs.append(_paragraphs[index])
        return paragraphs

    def get_n_words(self, query, doc_name):
        n_words = self.L - len(self.tokenize(query,  self.btokenize(query))) - 7 - 12
        return 10*math.floor((n_words-len(self.tokenize(doc_name, self.btokenize(doc_name))))/10.0)

    def get_contents_from_query(self, query, n_docs, only_first=False):
        doc_names = self.get_titles_from_query(query, n_docs)
        return [self.get_contents_from_title(doc_name,
                                             n_words=self.get_n_words(query, doc_name),
                                             only_first=only_first)
                        for doc_name in doc_names]

    def get_paragraphs_from_titles(self, query, doc_names, n_paragraphs, only_first,
                                   run_bm25=False):
        contents = []
        for doc_name in doc_names:
            contents += self.get_contents_from_title(doc_name,
                                                  n_words=self.get_n_words(query, doc_name),
                                                  only_first=only_first)
            if len(contents)>=n_paragraphs and not run_bm25:
                break
        if not run_bm25:
            return contents[:n_paragraphs]
        paragraphs = self.get_paragraphs_from_documents(query, contents, n_paragraphs,
                                                  only_first=only_first)
        return paragraphs #[:n_paragraphs]

    def get_paragraphs_from_query(self, query, n_docs, n_paragraphs, only_first=False):
        doc_names = self.get_titles_from_query(query, n_docs)
        return self.get_paragraphs_from_titles(query, doc_names, n_paragraphs,
                                               only_first=only_first, run_bm25=True)

    def get_paragraphs_from_keywords(self, query, keywords, n_paragraphs, only_first=True):
        doc_names = []
        for keyword in keywords:
            if type(keyword)==tuple:
                keyword, _ = keyword
            assert keyword in self.keyword2title
            for doc_name in self.keyword2title[keyword]:
                if doc_name not in doc_names:
                    doc_names.append(doc_name)
        return self.get_paragraphs_from_titles(query, doc_names, n_paragraphs, only_first=only_first)

    def get_keyword2title(self, keywords):
        keyword2title = defaultdict(list)
        for keyword in keywords:
            if type(keyword)==tuple:
                keyword, aliases = keyword
            else:
                aliases = []
            if keyword in keyword2title:
                continue
            if keyword in self.keyword2title:
                keyword2title[keyword] = self.keyword2title[keyword]
                continue
            if self.db.get_doc_text(keyword) is not None:
                keyword2title[keyword].append(keyword)
            else:
                for t in aliases:
                    if t!=keyword and self.db.get_doc_text(t) is not None:
                        keyword2title[keyword].append(t)
            if len(keyword2title[keyword])==0:
                doc = self.get_titles_from_query(keyword, 1)
                if len(doc)>0 and doc[0]!=keyword and doc[0] not in aliases and self.db.get_doc_text(doc[0]) is not None:
                    keyword2title[keyword].append(doc[0])
        self.keyword2title.update(keyword2title)
        return keyword2title

