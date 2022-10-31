#pragma once

/*
 * Copyright 2022 BDAP team.
 *
 * Author: Laurens Devos
 * Version: 0.1
 */

#include <unordered_map> // std::hash for std::string_view
#include "email.hpp"

namespace bdap {

/**
 * A base class for your classifiers.
 * Your implementations should extend this class as follows:
 *
 * ```
 *     class YourClf : public BaseClf<YourClf> {
 *         ...
 *     };
 * ```
 * Your class will fail to compile if it does not implement the methods
 *  - `update(const Email&)`
 *  - `predict(const Email&) const`
 *
 * You must follow this structure for ease of grading.
 *
 * This design pattern is called the 'curiously recurring template pattern'.
 *
 * You should not have to change this class. You do not have to submit this
 * class. If you find issues, contact your TA.
 */
template <typename Derived>
class BaseClf {
public:
    // Statistics
    int num_examples_processed = 0;

    // Classifiers parameters
    int ngram_k = 3;
    double threshold = 0.0;

    /** Update the paramters of the model using the incoming email (online
     * learning). */
    void update(const Email& email)
    {
        ++num_examples_processed;
        static_cast<Derived *>(this)->update_(email);
    }

    /** Use the current model to make a prediction about the given email. */
    double predict(const Email& email) const
    {
        return static_cast<const Derived *>(this)->predict_(email);
    }

    /** Threshold the prediction given by `predict` by `threshold` to get a
     * concrete classification. */
    bool classify(const Email& email) const
    { return classify(predict(email)); }

    bool classify(double pr) const
    { return pr > threshold; }

    /* UTILITY FUNCTIONS */

    static size_t hash(std::string_view key, size_t seed)
    {
        // use std::hash implementation of std::unordered_map
        size_t h = std::hash<std::string_view>{}(key);
        return (1000003 * h) ^ seed;
    }

    /* IMPLEMENT THESE METHODS IN YOUR SUBCLASSES */
    void update_(const Email& email);
    double predict_(const Email& email) const;
};

} // namespace bdap
