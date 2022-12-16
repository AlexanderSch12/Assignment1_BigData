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

void load_emails(std::vector<Email>& emails, const std::string& fname)
{
    std::ifstream f(fname);
    if (!f.is_open())
    {
        std::cerr << "Failed to open file `" << fname << "`, skipping..." << std::endl;
    }
    else
    {
        steady_clock::time_point begin = steady_clock::now();
        read_emails(f, emails);
        steady_clock::time_point end = steady_clock::now();

        std::cout << "Read " << fname << " in "
                  << (duration_cast<milliseconds>(end-begin).count()/1000.0)
                  << "s" << std::endl;
    }
}

std::vector<Email> load_emails(int seed)
{
    std::vector<Email> emails;

    // Windows
//    load_emails(emails, "C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\Enron.txt");
//    load_emails(emails,"C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\SpamAssasin.txt");
//    load_emails(emails,"C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\Trec2005.txt");
//    load_emails(emails, "C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\Trec2006.txt");
//    load_emails(emails, "C:\\Users\\alexa\\Documents\\KUL\\BigData\\Assignment1\\Assignment1_BigData\\data\\Trec2007.txt");

   // Remote Linux
   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/Enron.txt");
   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/SpamAssasin.txt");
   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/Trec2005.txt");
   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/Trec2006.txt");
   load_emails(emails, "/home/r0673385/Documents/BigData/Assignment1/Assignment1_BigData/data/Trec2007.txt");

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
template <typename Clf, typename Metric>
std::tuple<std::vector<double>,std::vector<double>,std::vector<double>>
stream_emails(const std::vector<Email> &emails,
              Clf& clf, Metric& metric, int window)
{
    std::vector<double> accuracy;
    std::vector<double> precision;
    std::vector<double> recall;
    for (size_t i = 0; i < emails.size(); i+=window)
    {
        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            metric.evaluate(clf, emails[i+u]);

        accuracy.push_back(metric.get_score());
        precision.push_back(metric.get_precision());
        recall.push_back(metric.get_recall());

        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            clf.update(emails[i+u]);
    }
    return std::make_tuple(accuracy,precision,recall);
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
    std::cout << "#emails: " << emails.size() << std::endl;

    Accuracy metric;
    NaiveBayesFeatureHashing bh{17,0.5};
    NaiveBayesCountMin bcm{3,17,0.5};
    PerceptronFeatureHashing ph{17, 0.8};
    PerceptronCountMin pcm{3,17,0.8};
    bh.ngram_k = 3;
    bcm.ngram_k = 3;
    ph.ngram_k = 3;
    pcm.ngram_k =3;

    steady_clock::time_point begin = steady_clock::now();
    auto [accuracy,precision,recall] = stream_emails(emails, bh, metric, 100);
    steady_clock::time_point end = steady_clock::now();

    std::cout << "------- Bayes Hashing ------- " << std::endl;
    std::cout << (duration_cast<milliseconds>(end-begin).count()/1000.0) << "s" << std::endl;
    std::cout << "Accuracy: " <<  accuracy[accuracy.size()-1] << std::endl;
    std::cout << "Precision: " << precision[precision.size()-1] << std::endl;
    std::cout << "Recall: " << recall[recall.size()-1] << std::endl;
    std::cout << std::endl;

    begin = steady_clock::now();
    auto [accuracy1,precision1,recall1] = stream_emails(emails, bcm, metric, 100);
    end = steady_clock::now();

    std::cout << "------- Bayes CountMin ------- " << std::endl;
    std::cout << (duration_cast<milliseconds>(end-begin).count()/1000.0) << "s" << std::endl;
    std::cout << "Accuracy: " <<  accuracy1[accuracy1.size()-1] << std::endl;
    std::cout << "Precision: " << precision1[precision1.size()-1] << std::endl;
    std::cout << "Recall: " << recall1[recall1.size()-1] << std::endl;
    std::cout << std::endl;

    begin = steady_clock::now();
    auto [accuracy2,precision2,recall2] = stream_emails(emails, ph, metric, 100);
    end = steady_clock::now();

    std::cout << "------- Peceptron Hashing ------- " << std::endl;
    std::cout << (duration_cast<milliseconds>(end-begin).count()/1000.0) << "s" << std::endl;
    std::cout << "Accuracy: " <<  accuracy2[accuracy2.size()-1] << std::endl;
    std::cout << "Precision: " << precision2[precision2.size()-1] << std::endl;
    std::cout << "Recall: " << recall2[recall2.size()-1] << std::endl;
    std::cout << std::endl;

    begin = steady_clock::now();
    auto [accuracy3,precision3,recall3] = stream_emails(emails, pcm, metric, 100);
    end = steady_clock::now();

    std::cout << "------- Perceptron CountMin ------- " << std::endl;
    std::cout << (duration_cast<milliseconds>(end-begin).count()/1000.0) << "s" << std::endl;
    std::cout << "Accuracy: " <<  accuracy3[accuracy3.size()-1] << std::endl;
    std::cout << "Precision: " << precision3[precision3.size()-1] << std::endl;
    std::cout << "Recall: " << recall3[recall3.size()-1] << std::endl;
    std::cout << std::endl;
    // write out the results
//    std::ofstream bh_acc{"bh_acc"};
//    std::ofstream bh_prec{"bh_prec"};
//    std::ofstream bh_rec{"bh_rec"};
//
//    bh_acc << "window=" << 125 << std::endl;
//    bh_acc << "log_buckers=" << 12 << std::endl;
//    bh_acc << "threshold=" << 0.8 << std::endl;
//
//    bh_prec << "window=" << 125 << std::endl;
//    bh_prec << "log_buckers=" << 12 << std::endl;
//    bh_prec << "threshold=" << 0.8 << std::endl;
//
//    bh_rec << "window=" << 125 << std::endl;
//    bh_rec << "log_buckers=" << 12 << std::endl;
//    bh_rec << "threshold=" << 0.8 << std::endl;
//    for (int i = 0 ; i < accuracy.size() ; i++)
//    {
//        bh_acc << accuracy[i] << std::endl;
//        bh_prec << precision[i] << std::endl;
//        bh_rec << recall[i] << std::endl;
//    }

    return 0;
}
