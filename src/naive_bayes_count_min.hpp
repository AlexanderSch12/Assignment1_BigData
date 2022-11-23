#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesCountMin : public BaseClf<NaiveBayesCountMin> {
    int log_num_buckets_;
    std::vector<double> buckets_; // First num_buckets are ham, rest num_buckets is spam
    std::vector<int> seeds_;
    int num_buckets_;
    double num_ngram_spam;
    double num_ngram_ham;
    double num_spam;
    double num_ham;
    int num_hashes_;
    // For different hash functions, the seed can be changed

public:
    NaiveBayesCountMin(int num_hashes, int log_num_buckets, double threshold)
    : log_num_buckets_(log_num_buckets), num_buckets_(1 << log_num_buckets),
    num_hashes_(num_hashes)
    {
        buckets_.resize(2*num_hashes_*num_buckets_,1);
        seeds_.resize(num_hashes_);

        for(int i = 0 ; i<num_hashes_ ; i++)
        {
            seeds_[i] = 0x9748cd * i;
        }
        num_ngram_spam = 1;
        num_ngram_ham = 1;
        num_spam = 1;
        num_ham = 1;
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
            offset = num_hashes_*num_buckets_;
        } else
        {
            num_ham++;
            num_ngram_ham += iter.size();
            offset = 0;
        }
        while (iter)
        {
            auto next = iter.next();
            for(int i = 0 ; i<num_hashes_ ; i++)
            {
                buckets_[offset + i*num_buckets_ + get_bucket(next,seeds_[i])]++;
            }
        }
    }

    double predict_(const Email& email) const
    {
        // TODO implement this
        return 0.0;
    }
private:
    size_t get_bucket(std::string_view ngram,int seed) const
    { return get_bucket(hash(ngram, seed)); }

    size_t get_bucket(size_t hash) const
    {
        hash = hash % num_buckets_;
        return hash;
    }
};

} // namespace bdap
