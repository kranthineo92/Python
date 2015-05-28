from os import listdir
from os.path import isfile, join, isdir
from math import log
from re import split
import sys
__author__ = 'KranthiDhanala'

#defining main


def main():
    spam_folder = sys.argv[1]
    ham_folder = sys.argv[2]
    test_spam_folder = sys.argv[3]
    test_ham_folder = sys.argv[4]
    stopwords_file = sys.argv[5]
    stop_words = []
    if isfile(stopwords_file):
        with open(stopwords_file) as stop_file:
            for line in stop_file:
                stop_words.append(line.strip())



    if not isdir(spam_folder):
        print 'Entered spam directory path is wrong!!please check'
        exit(1)

    if not isdir(ham_folder):
        print 'Entered spam directory path is wrong!!please check'
        exit(1)

    if not isdir(test_spam_folder):
        print "Entered test spam directory is not valid.please check"
        exit(1)

    if not isdir(test_ham_folder):
        print "Entered test ham directory is not valid.please check"
        exit(1)

    vocab_dict = {}

    no_of_spam_files, spam_word_dict, vocab_dict = file_word_counter(spam_folder, vocab_dict,stop_words)

    if no_of_spam_files == 0:
        print "entered spam directory does not have any files"
        exit(1)

    no_of_ham_files, ham_word_dict, vocab_dict = file_word_counter(ham_folder, vocab_dict,stop_words)

    if no_of_ham_files == 0:
        print "entered ham directory does not have any files"
        exit(1)

    prob_spam = log(float(no_of_spam_files)/(no_of_spam_files+no_of_ham_files), 10)
    prob_ham = log(float(no_of_ham_files)/(no_of_spam_files+no_of_ham_files), 10)
    vocab_length = len(vocab_dict)
    """
    calculate probability of each word given class (HAM/SPAM)
    """
    tot_no_words_spam = len(spam_word_dict)
    for key in spam_word_dict.keys():
        spam_word_dict[key] = log(float(spam_word_dict[key]+0.5)/(tot_no_words_spam+vocab_length), 10)

    tot_no_words_ham = len(ham_word_dict)
    for key in ham_word_dict.keys():
        ham_word_dict[key] = log(float(ham_word_dict[key]+0.5)/(tot_no_words_ham+vocab_length), 10)

    """
    classify new documents
    """

    (correctly_classified_spam, total_test_spam_files) = classify(test_spam_folder, spam_word_dict, ham_word_dict, vocab_length, prob_ham, prob_spam,stop_words, 0)
    (correctly_classified_ham, total_test_ham_files) = classify(test_ham_folder, spam_word_dict, ham_word_dict, vocab_length, prob_ham, prob_spam,stop_words, 1)
    print correctly_classified_spam, ' are classified correctly in spam folder out of',\
        total_test_spam_files, ' test spam files with accuracy of ', (float(correctly_classified_spam)/total_test_spam_files)*100

    print correctly_classified_ham, ' are classified correctly in ham folder out of',\
        total_test_ham_files, ' test ham files with accuracy of ', (float(correctly_classified_ham)/total_test_ham_files)*100


def classify(test_folder, spam_word_dict, ham_word_dict, vocab_length, prob_ham, prob_spam,stop_words, cls):
    spam_count = 0
    ham_count = 0
    test_files = [f_name for f_name in listdir(test_folder) if isfile(join(test_folder, f_name))]
    test_files_count = len(test_files)
    zero_prob_spam = log(float(0.5)/(len(spam_word_dict)+vocab_length))
    zero_prob_ham = log(float(0.5)/(len(ham_word_dict)+vocab_length))
    word_spam_counter = 0
    total_words = 0
    word_ham_counter = 0
    for test_file in test_files:
        with open(join(test_folder, test_file), 'r') as f:
            spam_prob_file = prob_spam
            ham_prob_file = prob_ham
            for line in f:
                words_in_line = split("\W+",line.strip())
                for word in words_in_line:
                    if word not in stop_words:
                        total_words += 1
                        if spam_word_dict.has_key(word):
                            spam_prob_file += spam_word_dict[word]
                            word_spam_counter += 1
                        else:
                            spam_prob_file += zero_prob_spam

                        if ham_word_dict.has_key(word):
                            ham_prob_file += ham_word_dict[word]
                            word_ham_counter += 1
                        else:
                            ham_prob_file += zero_prob_ham
            if spam_prob_file >= ham_prob_file:
                spam_count += 1
            else:
                ham_count += 1
    print spam_count, ham_count, test_files_count,word_spam_counter,word_ham_counter,total_words
    if cls == 0:
        return spam_count, test_files_count
    else:
        return ham_count, test_files_count


def file_word_counter(directory_path, vocab_dict,stop_words):
    word_dict = {}
    """storing all file names in directory into list
     as listdir() returns all contents if condition is placed to check if it is a file"""
    list_of_files = [f_name for f_name in listdir(directory_path) if isfile(join(directory_path, f_name))]
    file_count = len(list_of_files)
    """
    for each file check if dictionary already contains word then increment its value by 1 or else add it to dict
    and do similar check in vocabulary dictionary
    """
    for file_name in list_of_files:
        with open(join(directory_path, file_name), 'r') as f:
            for line in f:
                words_in_line = split("\W+", line.strip())
                for each_word in words_in_line:
                    if each_word not in stop_words:
                        if word_dict.has_key(each_word):
                            word_dict[each_word] += 1
                        else:
                            word_dict[each_word] = 1
                        if vocab_dict.has_key(each_word):
                            vocab_dict[each_word] += 1
                        else:
                            vocab_dict[each_word] = 1
    return file_count, word_dict, vocab_dict

if __name__ == '__main__':
    main()
