import pandas as pd
from pyserini.search import SimpleSearcher


class BM25:
""" BM25 implementation based on the Pyserini base BM25 implementation.
"""

    def __init__(self, index_loc, k1=0.9, b=0.4):
        self.searcher = SimpleSearcher(index_loc)
        self.searcher.set_bm25(k1, b)
    
    def search(self, query):
        """ Performs search on the given query and returns a list of relevant
        documents sorted on relevance.
        """
        result = self.searcher.search(query, k=1000)
        return list(map(lambda x: x.docid, result))

    def search_with_scores(self, query):
        """ Performs search on the given query and retrieves the relevant documents
        with scores in a sorted list based on relevance.
        """
        result = self.searcher.search(query, k=1000)
        return list(map(lambda x: [x.docid, x.score], result))
    
    def evaluate_queries_file(self, loc, out_loc):
        """ Evaluates a trec topic list and writes on the output location.
        """
        counter = 0
        queries = pd.read_csv(loc, header=None, names=['query_id', 'query'], sep='\t')
        results = pd.DataFrame(columns=['query_id', 'doc_id', 'rank', 'score'], index=range(len(queries)*1000))
        for _, row in queries.iterrows():
            print(counter)
            result = self.search_with_scores(row.query)
            for i in range(0, len(result)):
                # TODO: Should start at 0 or 1?
                r = [row.query_id, result[i][0], i, result[i][1]]
                results.loc[counter] = r
                counter += 1
        
        results.to_csv(out_loc, sep=' ', header=None, index=False)


# Evaluate the 200 msmarco test queries.
bm25 = BM25("anserini/indexes/msmarco-passage/lucene-index-msmarco")
testloc = "anserini/collections/msmarco-passage/msmarco-test2019-queries.tsv"
bm25.evaluate_queries_file(testloc, "msmarco-test.csv")
