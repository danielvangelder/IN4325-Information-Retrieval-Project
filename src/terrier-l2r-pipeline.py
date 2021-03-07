CORPUS_PATH = "collections/msmarco-passage"
COLLECTION = "trec-deep-learning-passages"
COLLECTION_ZIP_PATH = "collections/msmarco-passage/collectionandqueries.tar.gz"

import pyterrier as pt

if not pt.started():
  pt.init(mem=8000)


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

pipeline = pt.FeaturesBatchRetrieve(index, wmodel="BM25", features=["WMODEL:BM25","WMODEL:Tf", "WMODEL:PL2"])
# print(index.__dict__)

import pandas as pd
# Load topics as df: [qid, query]
# load qrels as df: [qid, docno, label]
def load_qrels_file(path):
    df = pd.read_csv(path, sep='\t', names=['qid','q0','docno','label'])
    del df['q0']
    return df

def load_topics_file(path):
    df = pd.read_csv(path, sep='\t', names=['qid','query'])
    return df

import lightgbm as lgb

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
    num_iterations=10)

bm25 = pt.BatchRetrieve(index, wmodel="BM25")

print('Loading train/validation topics and qrels')
train_topics = load_topics_file('collections/msmarco-passage/queries.train.tsv')
train_qrels = load_qrels_file('collections/msmarco-passage/qrels.train.tsv')
validation_topics = load_topics_file('collections/msmarco-passage/queries.dev.tsv')
validation_qrels = load_qrels_file('collections/msmarco-passage/qrels.dev.tsv')

print('Training LambdaMART pipeline')
lmart_l_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_l, form="ltr")
lmart_l_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)
# pt.Experiment(
#     [bm25, lmart_l_pipe],
#     test_topics,
#     test_qrels,
#     ["map"],
#     names=["BM25 Baseline", "LambdaMART (xgBoost)", "LambdaMART (LightGBM)" ]
# )