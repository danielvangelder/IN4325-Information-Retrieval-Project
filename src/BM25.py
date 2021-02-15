import math
import numpy as np

class BM25(Retriever):
    def __init__(df_corpus, k1, b):
        super.__init__(df_corpus)
        self.k1 = k1
        self.b = b


    def index():
        """ Defines the indexing step for BM25. For BM25 it is not necessary to comput the entire TF-IDF matrix.
        In this case the TF value for each term in a document can be calculated during the scoring step. It is, however,
        necessary to compute the IDF values for all terms beforehand. Hence, we compute the IDF dictionary in this function.

        In addition, the total amount of documents and average document length are calculated and stored.
        """
        self.corpus_size = df_corpus.size() # FIXME: find right size function
        idf = {}
        tf = {}
        docs = [] # TODO: iterate over df
        doc_lengths = np.array()
        # For each term t we find how many documents contain t and store that in tf[t]
        for doc in docs:
            processed = {}
            doc_lengths.append(len(doc))
            for term in doc:
                if term not in processed:
                    processed.add(term)
                    if term not in tf.keys():
                        tf[term] = 0
                    tf[term] += 1
        
        # Calc idf for each term
        for term in tf.keys():
            # IDF values can be negative, this should (maybe?) be handled
            idf[term] = math.log(( (self.corpus_size - tf[term] + 0.5) / (tf[term] + 0.5) ) + 1)
        
        self.avgdl = np.sum(doc_lengths) / self.corpus_size
        self.IDF = idf

    def score(document_id, query):
        """Calculate the (Okapi) BM25 score for the document in the corpus given the query. For details on the computation, refer to the 
        relevant Wikipedia article. 
        """
        doc = self.corpus[document_id] # FIXME: this is probably not correct
        score = 0
        for term in query:
            idf = this.IDF[term]
            tf = doc.count(term) # FIXME: verify whether this works
            numerator = tf * (self.k1 + 1)
            denominator = tf + (self.k1 * (1 - self.b + (self.b * (len(doc) / self.avgdl))))
            score += idf * (numerator / denominator)
        return score


    def rank_corpus(query, top_n = 1000):
        """Ranks the top_n (default: 1000) documents in the corpus for a query.
        """
        return None
    