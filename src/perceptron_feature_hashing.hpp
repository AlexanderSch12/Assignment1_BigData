#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronFeatureHashing : public BaseClf<PerceptronFeatureHashing>
{
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;
    int num_buckets_;

    int seed_;

public:
    PerceptronFeatureHashing(int log_num_buckets, double learning_rate)
            : log_num_buckets_(log_num_buckets), learning_rate_(learning_rate), bias_(0.0), seed_(0x9748cd),
              num_buckets_(1 << log_num_buckets)
    {
        // set all weights to zero
        weights_.resize(num_buckets_, 0.0);
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
            weights_[get_bucket(iter.next())] += learning_rate_ * (dn - yn);
        }

        bias_ +=  learning_rate_ * (dn - yn);
    }

    double predict_(const Email &email) const
    {
        EmailIter iter = EmailIter(email, this->ngram_k);
        double prediction = 0.0;
        while (iter)
        {
            prediction += weights_[get_bucket(iter.next())];
        }

        return prediction + bias_;
    }

    void print_weights() const
    {
        std::cout << "bias " << bias_ << std::endl;
        for (size_t i = 0; i < weights_.size(); ++i)
        {
            std::cout << "w" << i << " " << weights_[i] << std::endl;
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
