/*
 * Copyright 2022 BDAP team.
 *
 * Author: Laurens Devos
 * Version: 0.1
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "email.hpp"
#include "metric.hpp"
#include "base_classifier.hpp"

#include "naive_bayes_feature_hashing.hpp"
#include "perceptron_feature_hashing.hpp"
#include "naive_bayes_count_min.hpp"
#include "perceptron_count_min.hpp"

using namespace bdap;

using std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

void load_emails(std::vector<Email> &emails, const std::string &fname)
{
    std::ifstream f(fname);
    if (!f.is_open())
    {
        std::cerr << "Failed to open file `" << fname << "`, skipping..." << std::endl;
    } else
    {
        steady_clock::time_point begin = steady_clock::now();
        read_emails(f, emails);
        steady_clock::time_point end = steady_clock::now();

        std::cout << "Read " << fname << " in "
                  << (duration_cast<milliseconds>(end - begin).count() / 1000.0)
                  << "s" << std::endl;
    }
}

std::vector<Email> load_emails(int seed)
{
    std::vector<Email> emails;

    // Update these paths to your setup
    // Data can be found on the departmental computers in /cw/bdap/assignment1

    // Windows
    load_emails(emails, "C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\Enron.txt");
    load_emails(emails,
                "C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\SpamAssasin.txt");
    load_emails(emails,
                "C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\Trec2005.txt");
    load_emails(emails,
                "C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\Trec2006.txt");
    load_emails(emails,
                "C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\Trec2007.txt");

    // Linux
//    load_emails(emails, "/mnt/c/Users/alexa/Documents/KUL/BigData/Assignment1/Assignment1_BigData/data/Enron.txt");
//    load_emails(emails, "/mnt/c/Users/alexa/Documents/KUL/BigData/Assignment1/Assignment1_BigData/data/SpamAssasin.txt");
//    load_emails(emails, "/mnt/c/Users/alexa/Documents/KUL/BigData/Assignment1/Assignment1_BigData/data/Trec2005.txt");
//    load_emails(emails, "/mnt/c/Users/alexa/Documents/KUL/BigData/Assignment1/Assignment1_BigData/data/Trec2006.txt");
//    load_emails(emails, "/mnt/c/Users/alexa/Documents/KUL/BigData/Assignment1/Assignment1_BigData/data/Trec2007.txt");

    // Remote
//   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/Enron.txt");
//   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/SpamAssasin.txt");
//   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/Trec2005.txt");
//   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/Trec2006.txt");
//   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/Trec2007.txt");

    // Shuffle the emails
    std::default_random_engine g(seed);
    std::shuffle(emails.begin(), emails.end(), g);

    return emails;
}

/**
 * This function emulates a stream of emails. Every `window` examples, the
 * metric is evaluated and the score is recorded. Use the results of this
 * function to plot your learning curves.
 */
template<typename Clf, typename Metric>
std::vector<double>
stream_emails(const std::vector<Email> &emails,
              Clf &clf, Metric &metric, int window)
{
    std::vector<double> metric_values;
    //std::cout << emails.size() << std::endl;
    for (size_t i = 0; i < emails.size(); i += window)
    {
        for (size_t u = 0; u < window && i + u < emails.size(); ++u)
            metric.evaluate(clf, emails[i + u]);

        double score = metric.get_score();
        metric_values.push_back(score);

        for (size_t u = 0; u < window && i + u < emails.size(); ++u)
            clf.update(emails[i + u]);
    }
    return metric_values;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: ./bdap_assignment1 <window-size> <ngram_k> <output-file>"
                  << std::endl;
        return 1;
    }

    int window = std::atoi(argv[1]);
    int ngram_k = std::atoi(argv[2]);
    std::string outfname{argv[3]};

    if (window <= 0)
    {
        std::cerr << "Invalid window size " << window << std::endl;
        return 2;
    }

    if (ngram_k <= 0)
    {
        std::cerr << "Invalid ngram_k value " << ngram_k << std::endl;
        return 3;
    }

    int seed = 12;
    std::vector<Email> emails = load_emails(seed);

    double max = 0.0;
    double max_acc, max_prec, max_rec;
    double max_acc2, max_prec2, max_rec2;
    double max_acc3, max_prec3, max_rec3;
    int best_ngram, best_window, best_buckets = 0, best_hash = 0;
    int best_ngram2, best_window2, best_buckets2 = 0, best_hash2 = 0;
    int best_ngram3, best_window3, best_buckets3 = 0, best_hash3 = 0;
    double best_thresh, best_thresh2, best_thresh3 = 0.0;
    double best_acc, best_prec, best_rec = 0.0;
    int ngram_acc, window_acc, buckets_acc = 0, hash_acc = 0;
    int ngram_prec, window_prec, buckets_prec = 0, hash_prec = 0;
    int ngram_rec, window_rec, buckets_rec = 0, hash_rec = 0;
    double thresh_acc, thresh_prec, thresh_rec = 0.0;

    Accuracy accuracy;
    std::ofstream outfile{outfname};
                        NaiveBayesFeatureHashing clf(5,0.6);
                        //PerceptronFeatureHashing clf(buckets,0.15);

                        // TODO: logBuckets cannot be big for vector, log 30 is too big for bayes hashing because *2
                        //NaiveBayesCountMin clf(hash, buckets, thresh);
                        //PerceptronCountMin clf(hash,buckets,thresh);
                        clf.ngram_k = 3;
                        auto accuracy_values = stream_emails(emails, clf, accuracy, 20);

                        double acc = accuracy.get_accuracy();
                        double prec = accuracy.get_precision();
                        double rec = accuracy.get_recall();


    std::cout << std::endl;

    std::cout << "Accuracy: " << acc << std::endl;
    std::cout << "Precision: " << prec << std::endl;
    std::cout << "Recall: " << rec << std::endl;
    return 0;
}
