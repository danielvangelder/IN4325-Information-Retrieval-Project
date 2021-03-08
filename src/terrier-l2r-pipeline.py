from typing import Any, Union

CORPUS_PATH = "collections/msmarco-passage"
COLLECTION = "trec-deep-learning-passages"
COLLECTION_ZIP_PATH = "collections/msmarco-passage/collectionandqueries.tar.gz"
FEATURES_BATCH_N = 1000
RUN_ID = "00"
TOP_N_TRAIN = 200
RANDOM_FOREST = "RANDOMFOREST"
LAMBDAMART = "LAMBDAMART"

import pyterrier as pt
import pandas as pd
import numpy as np
import lightgbm as lgb
# import xgboost as xgb
import time
from sklearn.ensemble import RandomForestRegressor
import sys
import string


def main(algorithm=LAMBDAMART, feat_batch=FEATURES_BATCH_N, top_n_train=TOP_N_TRAIN, run_id=RUN_ID):

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
        print("Indexing MSMARCO passage ranking dataset")
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


    # Load topics as df: [qid, query]
    # load qrels as df: [qid, docno, label]
    def load_qrels_file(path):
        df = pd.read_csv(path, sep='\t', names=['qid','q0','docno','label'], dtype={'qid': str, 'q0': str, 'docno': str, 'label': np.int32})
        del df['q0']
        return df

    
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





    print('Loading train/validation topics and qrels')
    train_topics = load_topics_file('collections/msmarco-passage/queries.train.tsv')
    train_qrels = load_qrels_file('collections/msmarco-passage/qrels.train.tsv')
    validation_topics = load_topics_file('collections/msmarco-passage/queries.dev.small.tsv')
    validation_qrels = load_qrels_file('collections/msmarco-passage/qrels.dev.small.tsv')
    test_topics = load_topics_file('collections/msmarco-passage/msmarco-test2019-queries.tsv')


    print('Getting first {} train topics and corresponding qrels'.format(top_n_train))
    # TODO: not all queries here have qrels... Maybe filter on first 100 that have qrels?
    if top_n_train > 0:
        train_sub = train_topics[:top_n_train].copy()
        train_qrels_sub = filter_train_qrels(train_sub, train_qrels)
        validation_sub = validation_topics[:top_n_train].copy()
        validation_qrels_sub = filter_train_qrels(validation_sub, validation_qrels)
    else:
        train_sub = train_topics
        train_qrels_sub = train_qrels
        validation_sub = validation_topics
        validation_qrels_sub = validation_qrels
    # print(train_qrels_sub)





    ##############
    ## TRAINING ##
    ##############



    print('Setting up FeaturesBatchRetriever')


    pipeline = pt.FeaturesBatchRetrieve(index, wmodel="BM25", features=["WMODEL:BM25", "WMODEL:Tf", "WMODEL:PL2"]) % feat_batch

    #### LAMBDAMART
    print('Configuring Ranker...')
    # this configures LightGBM as LambdaMART 
    lmart_l = lgb.LGBMRanker(
        task="train",
        min_data_in_leaf=1,
        min_sum_hessian_in_leaf=100,
        max_bin=255,
        num_leaves=7,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5, 10],
        learning_rate= .1,
        importance_type="gain",
        # num_iterations=10,
        silent=False,
        n_jobs=-1)

    # lmart_x = xgb.sklearn.XGBRanker(objective='rank:ndcg',
    #       learning_rate=0.1,
    #       gamma=1.0,
    #       min_child_weight=0.1,
    #       max_depth=6,
    #       verbose=2,
    #       random_state=42)



    print('''\n
    ######################################
    ##### Training pipeline summary: #####
    ######################################

    Train Topics: {}
    Train Qrels: {}
    Validation topics: {}
    Validation Qrels: {}
    Amount of passage samples per query: {}

    ######################################

    '''.format(train_sub.shape[0], train_qrels_sub.shape[0], validation_sub.shape[0], validation_qrels_sub.shape[0], FEATURES_BATCH_N))

    start = time.time()
    print("Model output is not rendered to the terminal until after the run is finished...")
    if algorithm.upper() == LAMBDAMART:
        print('Training LambdaMART pipeline')

        # ltr_pipeline = pipeline >> pt.ltr.apply_learned_model(lmart_x, form="ltr")
        # ltr_pipeline.fit(train_sub, train_qrels_sub, validation_topics, validation_qrels)

        ltr_pipeline = pipeline >> pt.ltr.apply_learned_model(lmart_l, form="ltr")
        ltr_pipeline.fit_kwargs = {'verbose': 1}
        ltr_pipeline.fit(train_sub, train_qrels_sub, validation_sub, validation_qrels_sub)
        model_name = "LambdaRANK"

    elif algorithm.upper() == RANDOM_FOREST:
        # RANDOM FOREST
        print('Training RandomForest pipeline')

        ltr_pipeline = pipeline >> pt.ltr.apply_learned_model(RandomForestRegressor(n_jobs=-1,verbose=10))
        ltr_pipeline.fit(train_sub, train_qrels_sub, validation_sub, validation_qrels_sub)
        model_name = 'RandomForest'

    ### End of training ###

    end = time.time()
    print('Training finished, time elapsed:', end - start, 'seconds...')



    ###########################
    ## RERANKING AND OUTPUT  ##
    ###########################


    print('Running test evaluation...')

    # Test on small subset
    # res = ltr_pipeline.transform(test_topics[:10].copy())

    # Test on entire testset
    start = time.time()
    res = ltr_pipeline.transform(test_topics)
    end = time.time()
    print('Test evaluation finished, time elapsed:', end - start, 'seconds...')


    print('Writing results...')
    output_file_path = './{}_resuls_{}.trec'.format(model_name,str(run_id))
    pt.io.write_results(res,output_file_path,format='trec')
    print('SUCCES: results can be found at: ', output_file_path)


if __name__ == "__main__":
    file_name = sys.argv[0]
    if len(sys.argv) <= 1:
        main()
    elif len(sys.argv) == 5:
        main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    else:
        print("ERROR: supplied invalid amount of arguments, please pass either 0 or 4 arguments to the program.")
        print('''The arguments are: 
        - [algorithm]: either lambdamart or randomforest, default: lambdamart
        - [no. passages to retrieve in stage 1]: default: 1000
        - [amount of train/validation topics to use (values <= 0 will be interpreted as using all)], default: 100
        - [run name for file output]: default: 00''')
        sys.exit(1)



# bm25 = pt.BatchRetrieve(index, wmodel="BM25")

# pt.Experiment(
#     [bm25, lmart_l_pipe],
#     test_topics,
#     test_qrels,
#     ["map"],
#     names=["BM25 Baseline", "LambdaMART (xgBoost)", "LambdaMART (LightGBM)" ]
# )