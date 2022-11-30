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

    double signum(double a) const
    { return (a > 0) - (a < 0); }

    void update_(const Email &email)
    {
        EmailIter iter = EmailIter(email, this->ngram_k);

        // w(n+1) = w(n) + l[d(n) - y(n)]x(n)
        double yn = signum(predict_(email));
        int dn;
        if (email.is_spam()) dn = 1;
        else dn = -1;

        while (iter)
        {
            auto next = iter.next();
            for (int i = 0; i < num_hashes_; i++)
            {
                weights_[i * num_buckets_ + get_bucket(next, seeds_[i])] += learning_rate_ * (dn - yn);
            }
        }

        bias_ += learning_rate_ * (dn - yn);
    }

    double predict_(const Email &email) const
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        double prediction = 0.0;
        while (iter)
        {
            auto next = iter.next();
            double min = weights_[get_bucket(next, seeds_[0])];
            for (int i = 1; i < num_hashes_; i++)
            {
                // Find min
                double current_value = weights_[i * num_buckets_ + get_bucket(next, seeds_[i])];
                if (current_value < min)
                {
                    min = current_value;
                }
            }
            // Use smallest
            prediction += min;
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
