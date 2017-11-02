# Joshua Meyer (2017)

I've added this main script <read_corpus.py> to give an idea of how
to get raw text documents into the format required by
logistic_regression_train.py.

read_corpus.py expects data in this format:

data/
    ham/
        hamfile1.txt
        hamfile2.txt
        hamfile3.txt
        
    spam/
        spamfile1.txt
        spamfile2.txt
        spamfile3.txt


To give a real case, I made <format_SMS.sh> to format an easy to get
corpus, a dataset used in Kaggle <https://www.kaggle.com/uciml/sms-spam-collection-dataset>, originally from <https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection>.

So, if you can:

(1) download the SMS corpus
(2) format it with <format_SMS.sh>
(3) process new data/ dir with read_corpus.py

You should be able to process your own data with the same pipeline.

This is just one way to get features from text, and it makes very
sparse matrices, so be careful, you can use up a lot of RAM by training on a very sparse trainX.csv.


