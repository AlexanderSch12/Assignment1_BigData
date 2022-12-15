#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronCountMin : public BaseClf<PerceptronCountMin>
{
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;
    std::vector<int> seeds_;

    int num_buckets_;
    int num_hashes_;

public:
    PerceptronCountMin(int num_hashes, int log_num_buckets, double learning_rate)
            : log_num_buckets_(log_num_buckets), learning_rate_(learning_rate), bias_(0.0),
              num_buckets_(1 << log_num_buckets), num_hashes_(num_hashes)
    {
        weights_.resize(num_hashes_ * num_buckets_, 0.0);
        seeds_.resize(num_hashes_);

        seeds_[0] = 0x9748cd;
        for (int i = 1; i < num_hashes_; i++)
        {
            seeds_[i] = seeds_[i - 1] * i;
        }

    }

    static int signum(double a)
    { return (a > 0) - (a < 0); }

    void update_(const Email &email)
    {
        EmailIter iter = EmailIter(email, this->ngram_k);

        // w(n+1) = w(n) + l[d(n) - y(n)]x(n)
        int yn = signum(predict_(email));
        int dn;
        if (email.is_spam()) dn = 1;
        else dn = -1;

        int error = dn - yn;
        while (iter && error != 0)
        {
            auto next = iter.next();
            for (int hash = 0; hash < num_hashes_; hash++)
            {
                weights_[hash * num_buckets_ + get_bucket(next, seeds_[hash])] += learning_rate_ * error;
            }
        }

        bias_ += learning_rate_ * error;
    }

    double predict_(const Email &email) const
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        double prediction = 0.0;

        std::vector<double> median_weights(num_hashes_);

        while(iter)
        {
            auto next = iter.next();
            for (int i = 0; i < num_hashes_; i++)
            {
                median_weights[i] = weights_[i * num_buckets_ + get_bucket(next,seeds_[i])];
            }

            int n = median_weights.size();
            if (n % 2 == 0)
            {
                nth_element(median_weights.begin(), median_weights.begin() + n / 2, median_weights.end());
                nth_element(median_weights.begin(), median_weights.begin() + (n - 1) / 2, median_weights.end());
                prediction += (double) (median_weights[(n - 1) / 2] + median_weights[n / 2]) / 2.0;
            } else
            {
                nth_element(median_weights.begin(), median_weights.begin() + n / 2, median_weights.end());
                prediction += (double) median_weights[n / 2];
            }
        }
        return prediction + bias_;
    }

private:
    size_t get_bucket(std::string_view ngram, int seed) const
    { return get_bucket(hash(ngram, seed)); }

    size_t get_bucket(size_t hash) const
    {
        hash = hash % num_buckets_;
        return hash;
    }
};

} // namespace bdap
