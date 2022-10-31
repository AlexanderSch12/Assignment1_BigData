#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronCountMin : public BaseClf<PerceptronCountMin> {
    // TODO add private fields here

public:
    // TODO initialize your fields in the constructor
    PerceptronCountMin(int num_hashes, int log_num_buckets, double learning_rate) {}

    void update_(const Email& email)
    {
        // TODO implement this
    }

    double predict_(const Email& email) const
    {
        // TODO implement this
        return 0.0;
    }
};

} // namespace bdap
