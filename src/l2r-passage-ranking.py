import sys
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.base import hits_to_texts
from pyserini.search import SimpleSearcher
from tqdm import tqdm
OUTPUT_PATH = 'runs/passage_reranking_monot5.tsv'
INDEX_PATH = 'indexes/msmarco-passage/lucene-index-msmarco'
QUERIES_PATH = 'collections/msmarco-passage/msmarco-test2019-queries.tsv'

def output_to_csv(queries, rankings, file_path='runs/monot5.csv'):
    '''Desired output format: 'query_id', 'doc_id', 'rank', 'score'
    '''
    with open(file_path, 'w') as f:
        for (i,q) in enumerate(queries):
            q_rank = rankings[i]
            for (j,r) in enumerate(q_rank):
                f.write(str(q.id) + ' ' + str(r.metadata['docid']) + ' ' + str(j) + ' ' + str(r.score) + '\n')
                
def output_to_tsv(queries, rankings, file_path='runs/monot5.tsv'):
    '''Desired output format: 'query_id', 'doc_id', 'rank', 'score'
    '''
    with open(file_path, 'w') as f:
        for (i,q) in enumerate(queries):
            q_rank = rankings[i]
            for (j,r) in enumerate(q_rank):
                f.write(str(q.id) + '\t' + str(r.metadata['docid']) + '\t' + str(j) + ' ' + str(r.score) + '\n')


def main(output_path=OUTPUT_PATH, index_path=INDEX_PATH, queries_path=QUERIES_PATH):
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
    print('Ranking queries using BM25 (k=50)')
    queries_text = []
    for query in tqdm(queries):
        hits = searcher.search(query.text, k=50)
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
        output_to_tsv(queries, rankings, output_path)
    elif '.csv' in output_path:
        output_to_csv(queries, rankings, output_path)
    else:
        print('ERROR: invalid output file format provided, please use either .csv or .tsv. Exiting')
        sys.exit(1)
    print('SUCCESS: completed reranking, you may check the output at:', output_path)
    sys.exit(0)




if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1:
        print('ERROR: invalid amount of arguments provided')
        print('Please pass either 0 or 3 arguments. Format: \"python l2r-passage-ranking.py OUTPUT_PATH INDEX_PATH TEST_QUERIES_PATH \", if 0 arguments passed, defaults are used:')
        print('output path:', OUTPUT_PATH, 'index path:', INDEX_PATH, 'queries path:', QUERIES_PATH)
        sys.exit(1)
    else:
        main()