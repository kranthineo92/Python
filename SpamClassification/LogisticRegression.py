from numpy import random,dot,array,exp
from random import choice
from os.path import isfile,isdir,join
from os import listdir
from re import split
import matplotlib.pyplot as pl

__author__ = 'KranthiDhanala'

def main():
    train_spam_folder = sys.argv[1]
    train_ham_folder = sys.argv[2]
    test_spam_folder = sys.argv[3]
    test_ham_folder = sys.argv[4]
    stopwords_file = sys.argv[5]
    stopwords = []
    if isfile(stopwords_file):
        with open(stopwords_file) as stop_file:
            for line in stop_file:
                stopwords.append(line.strip())

    if not isdir(train_spam_folder):
        print 'Entered spam directory path is wrong!!please check'
        exit(1)

    if not isdir(train_ham_folder):
        print 'Entered spam directory path is wrong!!please check'
        exit(1)

    if not isdir(test_spam_folder):
        print "Entered test spam directory is not valid.please check"
        exit(1)

    if not isdir(test_ham_folder):
        print "Entered test ham directory is not valid.please check"
        exit(1)

    vocab = []
    vocab.append("X0")
    vocab = build_vocab(train_spam_folder,vocab,stopwords)
    vocab = build_vocab(train_ham_folder,vocab,stopwords)
    training_data = []
    training_data = build_train_data(train_spam_folder,vocab,training_data,0)
    training_data = build_train_data(train_ham_folder,vocab,training_data,1)
    no_of_iterations = 100

    no_of_features = len(vocab)
    learning_rate = 0.5
    lmda = 0
    print 'training data is build now implementing perceptron algo with ', no_of_iterations, 'iterations and learning rate of ', learning_rate
    weights = impl_LR(training_data,no_of_iterations,no_of_features,learning_rate,lmda)
    print "Learning Done.."
    no_correct_ham_files,tot_ham_files = classify(test_ham_folder,vocab,weights,1)
    no_correct_spam_files,tot_spam_files = classify(test_spam_folder,vocab,weights,0)

    #print weights
    print 'ON TEST DATASET'
    print no_correct_ham_files, 'ham files are correctly classified out of ', tot_ham_files, 'with accuracy of ', (float(no_correct_ham_files)/tot_ham_files)*100
    print no_correct_spam_files, 'spam files are correctly classified out of ', tot_spam_files, 'with accuracy of ', (float(no_correct_spam_files)/tot_spam_files)*100

    print "ON TRAINING DATA SET"
    no_correct_ham_files,tot_ham_files = classify(train_ham_folder,vocab,weights,1)
    no_correct_spam_files,tot_spam_files = classify(train_spam_folder,vocab,weights,0)
    print no_correct_ham_files, 'ham files are correctly classified out of ', tot_ham_files, 'with accuracy of ', (float(no_correct_ham_files)/tot_ham_files)*100
    print no_correct_spam_files, 'spam files are correctly classified out of ', tot_spam_files, 'with accuracy of ', (float(no_correct_spam_files)/tot_spam_files)*100

"""Classifying the test files using weights from perceptron """
def classify(folder,vocab,weights,actual_class):
    unit_sig = lambda x: 0 if x < 0 else 1
    no_of_correctly_classified = 0
    tot_of_files = 0
    list_of_files = [f_name for f_name in listdir(folder) if isfile(join(folder,f_name))]
    tot_of_files = len(list_of_files)
    for file_name in list_of_files:
        train_example = []
        for word in vocab:
            train_example.append(0)
        with open(join(folder,file_name),'r') as f:
            for line in f:
                words = split("\W+",line.strip())
                for word in words:
                    if word in vocab:
                        word_index = vocab.index(word)
                        train_example[word_index] += 1
        xi = array(train_example)
        result = dot(xi,weights)
        pred_class = unit_sig(result)
        difference = pred_class - actual_class
        if difference == 0:
            no_of_correctly_classified += 1

    return no_of_correctly_classified,tot_of_files


"""Logistic Regression implementation"""
def impl_LR(training_data,no_of_iterations,no_of_features,learning_rate,lmda):
    weights = [0 for i in range(no_of_features)]
    error = []
    sigmoid = lambda x: 1.0/(1.0+exp(-x))
    for i in xrange(no_of_iterations):
        for xi,expected in training_data:
            result = dot(xi,weights)
            sig = sigmoid(result)
            change = learning_rate * xi * (expected-sig)
            #penalize = learning_rate * lmda * weights
            difference = change# - penalize
            weights += difference

    return weights

"""Building the training data as matrices using numpy """
def build_train_data(folder,vocab,training_data,cls):
    list_of_files = [f_name for f_name in listdir(folder) if isfile(join(folder,f_name))]
    for file_name in list_of_files:
        train_example = []
        for word in vocab:
            train_example.append(0)
        with open(join(folder,file_name),'r') as f:
            for line in f:
                words = split("\W+",line.strip())
                for word in words:
                    if word in vocab:
                        word_index = vocab.index(word)
                        train_example[word_index] += 1
        training_data.append([array(train_example),cls])
    return training_data


"""Building vocabulary from all the spam files and ham files"""
def build_vocab(folder,vocab,stopwords):
    list_of_files = [f_name for f_name in listdir(folder) if isfile(join(folder,f_name))]
    for file_name in list_of_files:
        with open(join(folder,file_name),'r') as f:
            for line in f:
                words = split("\W+",line.strip())
                for word in words:
                    if word not in stopwords:
                        if word not in vocab:
                            vocab.append(word)
    return vocab

if __name__ == "__main__":
    main()
