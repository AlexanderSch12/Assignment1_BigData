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
    std::vector<int> buckets_; // First num_buckets are ham, rest num_buckets is spam

    // TODO add fields here
    int num_buckets_;
    int num_ngram_spam;
    int num_ngram_ham;
    int num_spam;
    int num_ham;
    int seed_;

public:
    NaiveBayesFeatureHashing(int log_num_buckets, double threshold)
            : log_num_buckets_(log_num_buckets), seed_(0x249cd), num_buckets_(1 << log_num_buckets),
              buckets_(2 * (1 << log_num_buckets), 1)
    {
        // TODO initialize the data structures here
        num_ngram_spam = 0;
        num_ngram_ham = 0;
        num_spam = 0;
        num_ham = 0;
        // Initialized buckets_ with 1 (Laplace estimates)

        this->threshold = threshold;
    }

    void update_(const Email &email)
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        while (iter)
        {
            if (email.is_spam())
            {
                buckets_[num_buckets_ + get_bucket(iter.next())]++;
                num_ngram_spam++;
            } else
            {
                buckets_[get_bucket(iter.next())]++;
                num_ngram_ham++;
            }
        }
        if (email.is_spam()) num_spam++;
        else num_ham++;
    }

    double predict_(const Email &email) const
    {
        double probSpam = prob(email, num_buckets_, num_ngram_spam, num_spam);
        double probHam = prob(email, 0, num_ngram_ham, num_ham);

        double probability = probSpam - log(probSpam) + log1p(exp(log(probHam) - log(probSpam)));
        return probability;
    }

    double prob(const Email &email, int offset, int num_ngram, int num_mail) const
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        double count = 0;
        while (iter)
        {
            count += log(buckets_[offset + get_bucket(iter.next())]);
        }
        count -= email.num_words() * log(num_ngram);

        count += num_mail - (log(num_ham) + log1p(exp(log(num_spam) - log(num_ham))));

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
