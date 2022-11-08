#include "email.hpp"

namespace bdap {

struct Accuracy
{
    int n = 0;
    int correct = 0;
    int FP = 0;
    int TP = 0;
    int FN = 0;

    template<typename Clf>
    void evaluate(const Clf &clf, const std::vector<Email> &emails)
    {
        for (const Email &email: emails)
            evaluate(clf, email);
    }

    template<typename Clf>
    void evaluate(const Clf &clf, const Email &email)
    {
        bool lab = email.is_spam();
        double pr = clf.predict(email);
        bool pred = clf.classify(pr);
        ++n;
        correct += static_cast<int>(lab == pred);
        TP += static_cast<int>((lab && pred));
        FP += static_cast<int>((!lab && pred));
        FN += static_cast<int>((lab && !pred));
    }

    double get_accuracy() const
    { return static_cast<double>(correct) / n; }

    double get_error() const
    { return 1.0 - get_accuracy(); }

    double get_score() const
    { return get_accuracy(); }

    double get_precision() const
    { return static_cast<double>(TP) / (TP + FP); }

    double get_recall() const
    { return static_cast<double>(TP) / (TP + FN); }
};

struct Precision
{
    int FP = 0;
    int TP = 0;

    template<typename Clf>
    void evaluate(const Clf &clf, const std::vector<Email> &emails)
    {
        for (const Email &email: emails)
            evaluate(clf, email);
    }

    template<typename Clf>
    void evaluate(const Clf &clf, const Email &email)
    {
        bool lab = email.is_spam();
        double pr = clf.predict(email);
        bool pred = clf.classify(pr);
        TP += static_cast<int>((lab && pred));
        FP += static_cast<int>((!lab && pred));
    }

    double get_precision() const
    { return static_cast<double>(TP) / (TP + FP); }

    double get_score() const
    { return get_precision(); }
};

struct Recall
{
    int FN = 0;
    int TP = 0;

    template<typename Clf>
    void evaluate(const Clf &clf, const std::vector<Email> &emails)
    {
        for (const Email &email: emails)
            evaluate(clf, email);
    }

    template<typename Clf>
    void evaluate(const Clf &clf, const Email &email)
    {
        bool lab = email.is_spam();
        double pr = clf.predict(email);
        bool pred = clf.classify(pr);
        TP += static_cast<int>((lab && pred));
        FN += static_cast<int>((lab && !pred));
    }

    double get_recall() const
    { return static_cast<double>(TP) / (TP + FN); }

    double get_score() const
    { return get_recall(); }
};

struct ConfusionMatrix
{

    template<typename Clf>
    void evaluate(const Clf &clf, const std::vector<Email> &emails)
    {
        for (const Email &email: emails)
            evaluate(clf, email);
    }

    template<typename Clf>
    void evaluate(const Clf &clf, Email &emails)
    {

    }

};


} // namespace bdap
