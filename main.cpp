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

    // Update these paths to your setup
    // Data can be found on the departmental computers in /cw/bdap/assignment1
    load_emails(emails, "data/Enron.txt");
    load_emails(emails, "data/SpamAssasin.txt");
    load_emails(emails, "data/Trec2005.txt");
    load_emails(emails, "data/Trec2006.txt");
    load_emails(emails, "data/Trec2007.txt");

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
std::vector<double>
stream_emails(const std::vector<Email> &emails,
              Clf& clf, Metric& metric, int window)
{
    std::vector<double> metric_values;
    for (size_t i = 0; i < emails.size(); i+=window)
    {
        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            metric.evaluate(clf, emails[i+u]);

        double score = metric.get_score();
        metric_values.push_back(score);

        for (size_t u = 0; u < window && i+u < emails.size(); ++u)
            clf.update(emails[i+u]);
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

    std::cout << "window:  " << window << std::endl;
    std::cout << "ngram_k: " << window << std::endl;
    std::cout << "outfile: " << outfname << std::endl;

    int seed = 12;
    std::vector<Email> emails = load_emails(seed);
    std::cout << "#emails: " << emails.size() << std::endl;

    Accuracy metric;
    PerceptronFeatureHashing clf{9, 0.5};
    clf.ngram_k = ngram_k;
    auto metric_values = stream_emails(emails, clf, metric, window);

    // write out the results
    std::ofstream outfile{outfname};
    outfile << "window=" << window << std::endl;
    outfile << "ngram_k=" << ngram_k << std::endl;
    outfile << "#emails=" << emails.size() << std::endl;
    for (double metric_value : metric_values)
        outfile << metric_value << std::endl;

    // just for fun, evaluate a single email:
    Email email1("EMAIL> label=1", "free try now");
    Email email2("EMAIL> label=1", "winner winner winner");

    std::cout << "classify(email1) = " << clf.classify(email1) << std::endl;
    std::cout << "classify(email2) = " << clf.classify(email2) << std::endl;

    return 0;
}
