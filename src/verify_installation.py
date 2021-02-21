from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
from pyserini.search import get_topics


def main():
    try:
        # Location of the generated index
        index_loc = "indexes/msmarco-passage/lucene-index-msmarco"

        # Create a searcher object
        searcher = SimpleSearcher(index_loc)
        # Set the active scorer to BM25
        searcher.set_bm25(k1=0.9, b=0.4)
        # Fetch 3 results for the given test query
        results = searcher.search('this is a test query', k=3)
        # For all results print the docid and the score
        expected = ['5578280', '2016011', '7004677']
        docids = [x.docid for x in results]
        if expected != docids:
            raise Exception('Test query results do not match expected:', expected, '(expecteD)', docids, '(actual)')
        # IndexReader can give information about the index
        indexer = IndexReader(index_loc)
        if indexer.stats()['total_terms'] != 352316036:
            raise Exception('There are an unexpected number of terms in your index set, perhaps something went wrong while downloading and indexing the dataset?')
        topics = get_topics("msmarco-passage-dev-subset")
        if topics == {}:
            raise Exception('Could not find msmarco-passage-dev-subset... Best approach is to retry indexing the dataset.')
        first_query = topics[list(topics.keys())[0]]['title']
        if first_query != "why do people grind teeth in sleep":
            raise Exception('Found a different first query than expected in the dataset. Did you download the right dataset?')
        # Using the pyserini tokenizer/stemmer/etc. to create queries from scratch
        # Using the pyserini tokenizer/stemmer/etc. to create queries from scratch
        query = "This is a test query in which things are tested. Found using www.google.com of course!"
        # Tokenizing in pyserini is called Analyzing
        output = indexer.analyze(query)
        if len(output) != 9:
            raise Exception('Tokenizer is not working correctly, something is probably wrong in Anserini. Perhaps try to install Anserini again.')
    except Exception as inst:
        print('ERROR: something went wrong in the installation')
        print(inst)
    else:
        print("INSTALLATION OK")

if __name__ == "__main__":
    main()