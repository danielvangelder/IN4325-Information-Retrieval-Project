# IN4325 Information Retrieval Project
##### `Authors:` Thomas Bos (4543408) & Daniël van Gelder (4551028), group 7

This is the repository for the implementation of the IR project for the TU Delft MSc course IN4325 Information Retrieval. In this project we implement two baselines for a Passage Ranking task and perform an analysis of the results.

## Prerequisites:
- Python 3.6+
- Pyserini (latest)
- Pandas (latest)
- Numpy (latest)
- autocorrect (latest)
- jupyter notebook (latest)
- Maven 3.3+
- Java 11 or higher
- PyTerrier
- Download the [TREC 2019 Deep Learning Track Passage Ranking Dataset](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019#passage-ranking-dataset) and place the following (extracted) files in the `src/data` folder:
    - Collection
    - Queries
    - All Test files

Activate the virtual environment in the `venv` folder to start a python environment with the appropriate dependencies.

## Installing Anserini/Pyserini/PyTerrier/pygaggle and the MS-MARCO dataset 
The steps for this installation have been retrieved from the anserini and pyserini documentation. This process has been tested on Mac OSX and ~~Windows~~. The package manager used is PyPI, using conda is _not_ recommended.

### Pyserini

1. Clone the [anserini repository](https://github.com/castorini/anserini) using git clone with the `--recurse-submodules` option, this also installs the `eval` subfolder (note: the following code uses SSH while HTTPS is also possible): 
```sh
git clone git@github.com:castorini/anserini.git --recurse-submodules
```  
Verify if the subfolders `eval` and `tools` are non-empty, otherwise make sure to download those manually.

2. Move all the contents of the `anserini` folder into the `src` folder of the current project.

3. Build the Anserini project using maven (tests can be skipped since we are only building):
```sh
mvn clean package appassembler:assemble -DskipTests
```

4. Build the `tools` directory as follows:
```sh
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
cd tools/eval/ndeval && make && cd ../../..
```

5. Install pyserini using PyPI:
```sh
pip install pyserini
```

### MS-MARCO

6. Download and extract the MS MARCO dataset for Passage Ranking. We will create a new directory for this (make sure to do this from the `src` folder):
```sh
mkdir -p collections/msmarco-passage

wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz -P collections/msmarco-passage

# Alternative mirror:
# wget https://www.dropbox.com/s/9f54jg2f71ray3b/collectionandqueries.tar.gz -P collections/msmarco-passage

tar xvfz collections/msmarco-passage/collectionandqueries.tar.gz -C collections/msmarco-passage
```
If desired, the checksum of the downloaded collection file `collectionandqueries.tar.gz` can be checked: it should have an MD5 checksum of `31644046b18952c1386cd4564ba2ae69`.

7. Download the test queries file and place it in the directory with the other collection files:

```sh
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz -P collections/msmarco-passage
```

### Indexing MS-MARCO for PySerini

8. Convert the MS MARCO `.tsv` collection into Anserini's jsonl files:
```sh
python tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path collections/msmarco-passage/collection.tsv \
 --output-folder collections/msmarco-passage/collection_jsonl
```

9. Now index these docs as a `JsonCollection` using Anserini (this may take a few minutes):
```sh
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 9 -input collections/msmarco-passage/collection_jsonl \
 -index indexes/lucene-index-msmarco-passage -storePositions -storeDocvectors -storeRaw
```

10. This should complete the installation process. Verify that everything is correct by running `verify_installation.py` in the `src` folder. This should print `INSTALLATION OK` if everything is working correctly. If not, please refer to the installation of [anserini](https://github.com/castorini/anserini), [pyserini](https://github.com/castorini/pyserini) and the following [doc](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md#data-prep) to debug.

### PyTerrier

11. Install PyTerrier using PyPI as follows: `pip install python-terrier`. This should install Terrier as well.

12. Also make sure to install LightGBM using `Homebrew` as follows: `brew install lightgbm` and consequently through PyPI: `pip install lightgbm`

13. Make sure that `scikit-learn` is installed: `pip install -U scikit-learn`

14. Indexing the MSMARCO dataset will be performed by the pipeline. If the dataset has previously been indexed, then it will reload the index.
### Running BM25
In order to run the BM25 algorithm, run `src/BM25_pyserini.py`. Make sure that the locations to the index and query files specified at the bottom are correct.

### pygaggle

15. Install via PyPI: `pip install pygaggle`

16. Clone the repo recursively such that the submodules are downloaded as well: `git clone --recursive https://github.com/castorini/pygaggle.git`

17. Move all the contents of the repository into the `src` folder.

18. Make sure all the `pygaggle` requirements are installed: `pip install -r requirements.txt`

19. Installation can be verified by opening and running the `src/pygaggle-reference.ipynb` notebook.

## Running the pipeline

### BM25

### LambdaMART

### T5




## Reading the analysis

The notebook used for the error analysis 