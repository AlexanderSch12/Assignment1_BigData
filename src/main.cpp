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
    std::cout << emails.size() << std::endl;
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

    int max = 0;
    int max_acc, max_prec, max_rec;
    int best_ngram, best_window, best_buckets;
    double best_thresh;

    Accuracy accuracy;
    std::ofstream outfile{outfname};
    for (int ngram = 1; ngram < 8; ngram++)
    {
        for (int win = 10; win < 161; win += 25)
        {
            for (int buckets = 5; buckets < 21; buckets += 5)
            {
                for (double thresh = 0.5; thresh < 0.9; thresh += 0.1)
                {
                    for (int hash = 2; hash < 11; hash += 4)
                    {
                        //NaiveBayesFeatureHashing clf(buckets,thresh);
                        //PerceptronFeatureHashing clf(buckets,0.15);

                        // TODO: logBuckets cannot be big for vector, log 30 is too big for bayes hashing because *2
                        NaiveBayesCountMin clf(10, buckets, thresh);
                        clf.ngram_k = ngram;
                        auto accuracy_values = stream_emails(emails, clf, accuracy, 20);

                        outfile << "--------- ngram: " << ngram << " window: " << win << " log_bucket: " << buckets
                                << " threshold: " << thresh << " ---------" << std::endl;
                        outfile << "Accuracy = " << accuracy.get_accuracy() << std::endl;
                        outfile << "Precision = " << accuracy.get_precision() << std::endl;
                        outfile << "Recall = " << accuracy.get_recall() << std::endl;

                        std::cout << "--------- ngram: " << ngram << " window: " << win << " log_bucket: " << buckets
                                  << " threshold: " << thresh << " ---------" << std::endl;
                        std::cout << "Accuracy = " << accuracy.get_accuracy() << std::endl;
                        std::cout << "Precision = " << accuracy.get_precision() << std::endl;
                        std::cout << "Recall = " << accuracy.get_recall() << std::endl;

                        int new_max = accuracy.get_accuracy() + accuracy.get_precision() + accuracy.get_recall();
                        if (max < new_max)
                        {
                            max = new_max;
                            max_acc = accuracy.get_accuracy();
                            max_prec = accuracy.get_precision();
                            max_rec = accuracy.get_recall();
                            best_ngram = ngram;
                            best_window = win;
                            best_buckets = buckets;
                            best_thresh = thresh;
                        }
                    }
                }
            }
        }
    }
    outfile << "############## Best: " << "ngram: " << best_ngram << " buckets: " << best_buckets << " best_window: "
            << best_window << " best_threshold: " << best_thresh << "##############" << std::endl;
    outfile << "Accuracy: " << max_acc << std::endl;
    outfile << "Precision: " << max_prec << std::endl;
    outfile << "Recall: " << max_rec << std::endl;
    return 0;
}
