import sys
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.base import hits_to_texts
from pyserini.search import SimpleSearcher
from tqdm import tqdm
OUTPUT_PATH = 'runs/passage_reranking_monot5.tsv'
INDEX_PATH = 'indexes/msmarco-passage/lucene-index-msmarco'
QUERIES_PATH = 'collections/msmarco-passage/msmarco-test2019-queries.tsv'
RUN = 'MonoT5'
K = 100

def output_to_csv(queries, rankings, run, file_path):
    '''Desired output format: 'query_id', 'Q0', 'doc_id', 'rank', 'score', 'run name'
    '''
    with open(file_path, 'w') as f:
        for (i,q) in enumerate(queries):
            q_rank = rankings[i]
            for (j,r) in enumerate(q_rank):
                f.write(str(q.id) + ',Q0,' + str(r.metadata['docid']) + ',' + str(j) + ',' + str(r.score) + ',' + str(run) + '\n')
                
def output_to_tsv(queries, rankings, run, file_path):
    '''Desired output format: 'query_id', 'Q0', 'doc_id', 'rank', 'score', 'run name'
    '''
    with open(file_path, 'w') as f:
        for (i,q) in enumerate(queries):
            q_rank = rankings[i]
            for (j,r) in enumerate(q_rank):
                f.write(str(q.id) + '\tQ0\t' + str(r.metadata['docid']) + '\t' + str(j) + ' ' + str(r.score) + '\t' + str(run) + '\n')


def main(output_path=OUTPUT_PATH, index_path=INDEX_PATH, queries_path=QUERIES_PATH, run=RUN, k=K):
    print('################################################')
    print("##### Performing Passage Ranking using L2R #####")
    print('################################################')
    print("Output will be placed in:", output_path, ", format used will be TREC")
    print('Loading pre-trained model MonoT5...')
    from pygaggle.rerank.transformer import MonoT5
    reranker =  MonoT5()

    print('Fetching anserini-like indices from:', index_path)
    # fetch some passages to rerank from MS MARCO with Pyserini (BM25)
    searcher = SimpleSearcher(index_path)
    print('Loading queries from:', queries_path)
    with open(queries_path, 'r') as f:
        content = f.readlines()
        content = [x.strip().split('\t') for x in content] 
        queries = [Query(x[1], x[0]) for x in content]
    print(f'Ranking queries using BM25 (k={k})')
    queries_text = []
    for query in tqdm(queries):
        hits = searcher.search(query.text, k=K)
        texts = hits_to_texts(hits)
        queries_text.append(texts)
    
    print('Reranking all queries using MonoT5!')
    rankings = []

    for (i,query) in enumerate(tqdm(queries)):
        reranked = reranker.rerank(query, queries_text[i])
        reranked.sort(key=lambda x: x.score, reverse=True)
        rankings.append(reranked)   
    
    print('Outputting to file...')
    if '.tsv' in output_path:
        output_to_tsv(queries, rankings, run, output_path)
    elif '.csv' in output_path:
        output_to_csv(queries, rankings, run, output_path)
    else:
        print('ERROR: invalid output file format provided, please use either .csv or .tsv. Exiting')
        sys.exit(1)
    print('SUCCESS: completed reranking, you may check the output at:', output_path)
    sys.exit(0)




if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) > 1:
        print('ERROR: invalid amount of arguments provided')
        print('Please pass either 0 or 4 arguments. Format: \"python t5-passage-ranking.py OUTPUT_PATH INDEX_PATH TEST_QUERIES_PATH RUN\", if 0 arguments passed, defaults are used:')
        print('output path:', OUTPUT_PATH, 'index path:', INDEX_PATH, 'queries path:', QUERIES_PATH, 'run:', RUN)
        sys.exit(1)
    else:
        main()