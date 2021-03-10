import time
from math import log

import pandas as pd
import spacy
from autocorrect import Speller
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher

spell = Speller(lang='en')

class BM25:
    """ BM25 implementation based on the Pyserini base BM25 implementation.
    """

    def __init__(self, index_loc, k1=0.9, b=0.4, k=1000):
        self.searcher = SimpleSearcher(index_loc)
        self.searcher.set_bm25(k1, b)
        self.index_loc = index_loc
        self.k = k
    
    def search(self, query):
        """ Performs search on the given query and returns a list of relevant
        documents sorted on relevance.
        """
        result = self.searcher.search(query, k=self.k)
        return list(map(lambda x: x.docid, result))

    def search_with_scores(self, query):
        """ Performs search on the given query and retrieves the relevant documents
        with scores in a sorted list based on relevance.
        """
        result = self.searcher.search(query, k=self.k)
        return list(map(lambda x: [x.docid, x.score], result))
    
    def evaluate_queries_file(self, loc, out_loc, trec=False, relevance_weighing=False, 
        spell_checking=False, n_extra_terms=10):
        """ Evaluates a trec topic list and writes on the output location.
        """
        start_time = time.time()
        counter = 0
        spell_checked = 0
        queries = pd.read_csv(loc, header=None, names=['query_id', 'query'], sep='\t')
        results = pd.DataFrame(columns=['query_id', 'doc_id', 'rank', 'score'], index=range(len(queries)*1000))
        for _, row in queries.iterrows():
            print(counter)
            query = row.query
            if (spell_checking):
                old = query
                query = self.spell_check_query(query)
                if (query != old):
                    spell_checked += 1
                    print(spell_checked, ":", old, "->", query)
            if (relevance_weighing):
                print("OLD:", query)
                query = self.expand_query(query, 50, n_extra_terms)
                print(query)
            result = self.search_with_scores(query)
            for i in range(0, len(result)):
                r = [row.query_id, result[i][0], i, result[i][1]]
                results.loc[counter] = r
                counter += 1
        
        # Writes in TREC format
        if trec:
            results.insert(4, 'bm25', 'bm25')
            results.insert(1, 'Q0', 'Q0')
            results.score = results.score.apply(lambda x: round(x, 2))

        results.to_csv(out_loc, sep=' ', header=None, index=False)

        print("Time taken: {}".format(time.time() - start_time))

    def expand_query(self, query, k, n_extra_terms):
        """ Performs query expansion using Robertsons (1994) relevance weighing.
        """
        ranking = self.search(query)
        terms = {}
        index = IndexReader(self.index_loc)

        # Find all terms in the ranking
        for doc_id in ranking:
            doc_vec = index.get_document_vector(str(doc_id))
            for term in doc_vec.keys():
                stems = index.analyze(term)
                if len(stems) == 0:
                    continue
                stem = stems[0]
                if stem not in terms:
                    terms[stem] = 0
                terms[stem] += 1

        # Remove the terms already in the query
        for w in query.split(" "):
            try:
                del terms[index.analyze(w)[0]]
            except Exception:
                continue

        ows = {}
        stats = index.stats()
        N = stats['documents']
        R = self.k

        # Calculate the offer weights
        for term in terms:
            try:
                [n, _] = index.get_term_counts(term)
            except Exception:
                continue
            r = terms[term]
            ow = self.offer_weight(r, R, n, N)
            ows[term] = ow
        
        ows_list = [(k, v) for k, v in ows.items()]
        ows_list.sort(key=lambda x: x[1], reverse=True)

        expanded_query = query

        # Expand the query
        for i in range(n_extra_terms):
            expanded_query += " {}".format(ows_list[i][0])

        return expanded_query
        
    def offer_weight(self, r, R, n, N):
        """ Calculates offer weights according to Robertson, 1994
        """
        base = ( (r + 0.5) * (N - n - R + r + 0.5) ) / ( (n - r + 0.5) * (R - r + 0.5) )
        if base < 0:
            base = 0.001
        return r * log(base)

    def spell_check_query(self, query):
        """ Performs spell checking on a query
        """
        return " ".join(map(lambda w: spell(w), query.split(" ")))


# Evaluate the 200 msmarco test queries.
bm25 = BM25("../anserini/indexes/msmarco-passage/lucene-index-msmarco")
testloc = "../anserini/collections/msmarco-passage/msmarco-test2019-queries.tsv"
bm25.evaluate_queries_file(testloc, "bm25-msmarco-query-expansion-Test-.trec", trec=True, relevance_weighing=True, spell_checking=False, n_extra_terms=10)
