#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronFeatureHashing : public BaseClf<PerceptronFeatureHashing> {
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;

    int seed_;

public:
    PerceptronFeatureHashing(int log_num_buckets, double learning_rate)
        : log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0x9748cd)
    {
        // set all weights to zero
        weights_.resize(1 << log_num_buckets_, 0.0);
    }

    void update_(const Email& email)
    {
        // TODO implement this
    }

    double predict_(const Email& email) const
    {
        // TODO implement this
        return 0.0;
    }

    void print_weights() const
    {
        std::cout << "bias " << bias_ << std::endl;
        for (size_t i = 0; i < weights_.size(); ++i)
        {
            std::cout << "w" <<i << " " << weights_[i] << std::endl;
        }
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const
    {
        // TODO limit the range of the hash values here
        return hash;
    }
};

} // namespace bdap
