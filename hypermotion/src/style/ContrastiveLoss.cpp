#include "HyperMotion/style/ContrastiveLoss.h"

namespace hm::style {

ContrastiveLoss::ContrastiveLoss(float temperature)
    : temperature_(temperature) {}

ContrastiveLoss::~ContrastiveLoss() = default;

torch::Tensor ContrastiveLoss::compute(const torch::Tensor& embeddings) {
    // embeddings: [2*N, dim], pairs (2i, 2i+1) are positive
    int twoN = embeddings.size(0);
    int N = twoN / 2;

    // Cosine similarity matrix
    auto normed = torch::nn::functional::normalize(embeddings,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto sim = torch::mm(normed, normed.t()) / temperature_;

    // Mask out self-similarity
    auto mask = torch::eye(twoN, embeddings.options()).to(torch::kBool);
    sim.masked_fill_(mask, -1e9f);

    // For each row i, the positive is at index i^1 (XOR to get pair)
    auto labels = torch::arange(twoN, torch::TensorOptions().dtype(torch::kLong));
    for (int i = 0; i < twoN; ++i) {
        labels[i] = i ^ 1;  // XOR: 0<->1, 2<->3, etc.
    }
    labels = labels.to(embeddings.device());

    auto loss = torch::nn::functional::cross_entropy(sim, labels);
    return loss;
}

torch::Tensor ContrastiveLoss::computePairwise(
    const torch::Tensor& anchor, const torch::Tensor& positive) {
    // Concatenate and use the standard NT-Xent
    auto combined = torch::cat({anchor, positive}, 0);

    int N = anchor.size(0);
    int twoN = 2 * N;

    // Interleave: [a0, p0, a1, p1, ...]
    auto interleaved = torch::zeros_like(combined);
    for (int i = 0; i < N; ++i) {
        interleaved[2 * i] = anchor[i];
        interleaved[2 * i + 1] = positive[i];
    }

    return compute(interleaved);
}

} // namespace hm::style
