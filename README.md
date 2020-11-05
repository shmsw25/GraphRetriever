# GraphRetriever

This contains codes for the GraphRetriever that is proposed in the paper, [Knowledge Guided Text Retrieval and Reading for Open Domain Question Answering](https://arxiv.org/abs/1911.03868).

```
@article{ min2019knowledge ,
  title={ Knowledge Guided Text Retrieval and Reading for Open Domain Question Answering },
  author={ Min, Sewon and Chen, Danqi and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
  journal={ arXiv preprint arXiv:1911.03868 },
  year={ 2019 }
}
```

This README only describe the minimal set of command lines to run the GraphRetriever, and does not contain sufficient details about the model and how the code works. For more details, we recommend to read the paper or read the code.

## 0. Download QA data

Download the data for your QA task in `data/`.

```
mkdir data
wget https://nlp.cs.washington.edu/ambigqa/data/{webquestions|nq|triviaqa}.zip
unzip {webquestions|nq|triviaqa}.zip -d data/
rm {webquestions|nq|triviaqa}.zip
```

## 1. Preprocessing

For Wikipedia, you need DB and TF-IDF index by following [DrQA](https://github.com/facebookresearch/DrQA).

For WikiData, please run the following command.
```
python3 WikiData.py --dump_path {path_to_wikidata_dump} --data_dir {dir_for_saving_preprocessed_wikidata} --n_processed {n_processes_for_preprocessing}
```

This preprocessing code looks complicated, but what it really does is to store all the entities (in the DB called `Entities`), all the relations (in the DB called `Properties`) and all the triples (entity, relation, entity) (in the DB called `Claims`). We also store `Text2Entity` and `Text2Entity_norm` for mapping between entities and their text forms.

Running preprocessing may take more than a few days depending on multiprocess availability. We recommend to modify the code if you want to print out the progress during preprocessing.


## 2. Extracting entities from the question

Now, run `Tagme.py` to extract entities from the question.

```
python3 Tagme.py --data {webquestions|nq|triviaqa} --data_type {train|dev|test} --gcube_token {your_gcube_token}
```

You need GCUBE token in order to get an access to TAGME. Please refer to [here](https://pypi.org/project/tagme/) for details.

Running `Tagme.py` will save entity data in `data/{webquestions|nq|triviaqa}|{webquestions|nq|triviaqa}-{train|dev|test}.tagme.json`.

This part can be easily replaced by any better entity extraction model. For instance, you can use [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq) which has shown much better performance than TAGME on entity linking for questions (see [paper](https://arxiv.org/abs/2010.02413) for comparisons).

## 3. Running GraphRetriever

Now, in order to actually run the GraphRetriever to get a paragraph graph using entities in the question, Wikipedia and Wikidata, please run the following command.

```
python3 retrieve_hybrid.json --data {webquestions|nq|triviaqa} --data_type {train|dev|test} \
  --wiki_db_path {path_to_wiki_db_from_drqa} \
  --tfidf_path {path_to_tfidf_from_drqa} \
  --data_dir {path_to_wikidata_dir_you_preprocessed}
```

It will save retrieved paragraphs (along with the paragraph graph) in `data/{webquestions|nq|triviaqa}/{webquestions|nq|triviaqa}-{train|dev|test}.retrieved.json`.

The format of the saved data is as follows. Each line contains
- `question`: a string
- `answers`: a list of strings, containing all acceptable answers
- `paragraphs`: a list of paragraph, where each paragraph is represented by (title, index, tokenized context). "title" is the title of the Wikipedia page that this paragraph originated from, "index" is the ordering of this particular paragraph in the page, and "tokenized context" is a list of tokens based on BERTTokenizer.
- `graph`: a dictionary representing relationships between paragraphs. Each key is a string representing a paragraph pair separated by a space (e.g. `0 2` means a pair of 0-th paragraph and 2-th paragraph according to the list in `paragraphs`), and each value is a relation. It is either a dictionary containing `texts` and `description` if it is a cross-document relation (Wikidata relation), or `<CHILD_PARAGRAPH>`/`<PARENT_PARAGRAPH>` if it is an inner-document relation.

Note on tokenization:
- We did tokenization during retrieval because we set each paragraph to contain 300 tokens at maximum.
- Tokenization actually makes retrieval slow, so we implemented a caching behavior in `retriever.py`.
- Our BERTTokenizer is based on an older version of Huggingface transformers, inside `pytorch_transformers` directory. If you prefer, you can remove this directory and import tokenizers from an updated version of Huggingface transformers.
- If you prefer not to tokenize or want to use a simpler tokenization like `.split()`, please modify `retriever.py`.




