#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include <memory>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing>
{
    int log_num_buckets_;
    std::vector<double> buckets_; // First num_buckets are ham, rest num_buckets is spam

    int num_buckets_;
    double num_ngram_spam;
    double num_ngram_ham;
    double num_spam;
    double num_ham;
    int seed_;

public:
    NaiveBayesFeatureHashing(int log_num_buckets, double threshold)
            : log_num_buckets_(log_num_buckets), seed_(0x249cd), num_buckets_(1 << log_num_buckets),
              buckets_(2 * (1 << log_num_buckets), 1)
    {
        num_ngram_spam = 1;
        num_ngram_ham = 1;
        num_spam = 1;
        num_ham = 1;

        // Initialized buckets_ with 1 (Laplace estimates)

        this->threshold = threshold;
    }

    void update_(const Email &email)
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        int offset;
        if (email.is_spam())
        {
            num_spam++;
            num_ngram_spam += iter.size();
            offset = num_buckets_;
        } else
        {
            num_ham++;
            num_ngram_ham += iter.size();
            offset = 0;
        }
        while (iter)
        {
            buckets_[offset + get_bucket(iter.next())]++;
        }
    }

    double predict_(const Email &email) const
    {
        double probSpam = prob(email, num_buckets_, num_ngram_spam, num_spam);
        double probHam = prob(email, 0, num_ngram_ham, num_ham);

        double probability = probSpam - (probSpam + std::log1p(exp(probHam - probSpam)));
        return std::exp(probability);
    }

    double prob(const Email &email, int offset, double num_ngram, double num_mail) const
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        double count = 0;
        while (iter)
        {
            count += (std::log(buckets_[offset + get_bucket(iter.next())]));
        }
        count -= (iter.size() * log(num_ngram));
        count += (std::log(num_mail) - (std::log(num_ham) + log1p(exp(std::log(num_spam) - std::log(num_ham)))));
        return count;
    }

    void print_weights() const
    {
        for (size_t i = 0; i < num_buckets_; ++i)
        {
            std::cout << "w" << i << " " << buckets_[i] << ", " << buckets_[num_buckets_ + i] << std::endl;
        }
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const
    {
        hash = hash % num_buckets_;
        return hash;
    }
};

} // namespace bdap
