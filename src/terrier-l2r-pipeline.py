CORPUS_PATH = "collections/msmarco-passage"
COLLECTION = "trec-deep-learning-passages"
COLLECTION_ZIP_PATH = "collections/msmarco-passage/collectionandqueries.tar.gz"

import pyterrier as pt

if not pt.started():
  pt.init(mem=8000)



################
## INDEX STEP ##
################



dataset = pt.get_dataset("trec-deep-learning-passages")
def msmarco_generate():
    with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {'docno' : docno, 'text' : passage}

        

try:
    # Single threaded indexing           
    # iter_indexer = pt.IterDictIndexer("./passage_index")
    # indexref3 = iter_indexer.index(msmarco_generate(), meta=['docno', 'text'], meta_lengths=[20, 4096])            

    # Multi threaded indexing, UNIX-based systems only!!!!!   
    iter_indexer = pt.IterDictIndexer("./passage_index_8", threads=8)
    indexref4 = iter_indexer.index(msmarco_generate(), meta=['docno', 'text'], meta_lengths=[20, 4096])

except ValueError as err:
    if "Index already exists" in str(err):
        print("Index already exists, loading existing one")
        indexref4 = "./passage_index_8/data.properties"

pt.logging('WARN')
index = pt.IndexFactory.of(indexref4)
print(index.getCollectionStatistics().toString())



################
## DATA PREP  ##
################



pipeline = pt.FeaturesBatchRetrieve(index, wmodel="BM25", features=["WMODEL:BM25", "WMODEL:Tf", "WMODEL:PL2"])

import pandas as pd
import numpy as np
# Load topics as df: [qid, query]
# load qrels as df: [qid, docno, label]
def load_qrels_file(path):
    df = pd.read_csv(path, sep='\t', names=['qid','q0','docno','label'], dtype={'qid': str, 'q0': str, 'docno': str, 'label': np.int32})
    del df['q0']
    return df

import string
def load_topics_file(path):
    df = pd.read_csv(path, sep='\t', names=['qid','query'], dtype={'qid':str, 'query':str})
    exclude = set(string.punctuation)
    # Remove punctuation
    # print(exclude)
    df['query'] = df['query'].apply(lambda s: ''.join(ch for ch in s if ch not in exclude))
    # print(df['query'][:6])
    return df


def filter_train_qrels(train_topics_subset, train_qrels):
    m = train_qrels.qid.isin(train_topics_subset.qid)
    return train_qrels[m]


import lightgbm as lgb


print('Loading train/validation topics and qrels')
train_topics = load_topics_file('collections/msmarco-passage/queries.train.tsv')
train_qrels = load_qrels_file('collections/msmarco-passage/qrels.train.tsv')
validation_topics = load_topics_file('collections/msmarco-passage/queries.dev.small.tsv')
validation_qrels = load_qrels_file('collections/msmarco-passage/qrels.dev.small.tsv')
test_topics = load_topics_file('collections/msmarco-passage/msmarco-test2019-queries.tsv')


TOP_N_TRAIN = 10
print('Getting first {} train topics and corresponding qrels'.format(TOP_N_TRAIN))
# TODO: not all queries here have qrels... Maybe filter on first 100 that have qrels?
train_sub = train_topics[:TOP_N_TRAIN].copy()
train_qrels_sub = filter_train_qrels(train_sub, train_qrels)
validation_sub = validation_topics[:TOP_N_TRAIN].copy()
validation_qrels_sub = filter_train_qrels(validation_sub, validation_qrels)
# print(train_qrels_sub)


print('''Training/validation data sizes (rows):
Train Topics: {}
Train Qrels: {}
Validation topics: {}
Validation Qrels: {}
'''.format(train_sub.shape[0], train_qrels_sub.shape[0], validation_sub.shape[0], validation_qrels_sub.shape[0]))



##############
## TRAINING ##
##############

import time
start = time.time()

#### LAMBDAMART

# this configures LightGBM as LambdaMART 
lmart_l = lgb.LGBMRanker(task="train",
    min_data_in_leaf=1,
    min_sum_hessian_in_leaf=100,
    max_bin=255,
    num_leaves=7,
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[1, 3, 5, 10],
    learning_rate= .1,
    importance_type="gain",
    num_iterations=10,
    n_jobs=8)
import xgboost as xgb

lmart_x = xgb.sklearn.XGBRanker(objective='rank:ndcg',
      learning_rate=0.1,
      gamma=1.0,
      min_child_weight=0.1,
      max_depth=6,
      verbose=2,
      random_state=42)

print('Training LambdaMART pipeline')

lmart_l_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_x, form="ltr")
lmart_l_pipe.fit(train_sub, train_qrels_sub, validation_topics, validation_qrels)

# lmart_l_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_l, form="ltr")
# lmart_l_pipe.fit(train_sub, train_qrels_sub, validation_topics, validation_qrels)

## RANDOM FOREST
# print('Training RandomForest pipeline')
# from sklearn.ensemble import RandomForestRegressor

# random_forest_pipe = pipeline >> pt.ltr.apply_learned_model(RandomForestRegressor(n_estimators=1))
# random_forest_pipe.fit(train_sub, train_qrels_sub, validation_sub, validation_qrels_sub)
end = time.time()
print('Training finished, time elapsed:', end - start, 'seconds...')



###########################
## RERANKING AND OUTPUT  ##
###########################


print('Running test evaluation...')
res = lmart_l_pipe.transform(test_topics)
# res = random_forest_pipe.transform(test_topics)
# FIXME: res????
print('Writing results...')
pt.io.write_results(res,'./Randomforest_resuls.trec',format='trec')
print('DONE')
# bm25 = pt.BatchRetrieve(index, wmodel="BM25")

# pt.Experiment(
#     [bm25, lmart_l_pipe],
#     test_topics,
#     test_qrels,
#     ["map"],
#     names=["BM25 Baseline", "LambdaMART (xgBoost)", "LambdaMART (LightGBM)" ]
# )