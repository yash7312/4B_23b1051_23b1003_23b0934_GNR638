#include "tensor.hpp"
#include <stdexcept>
#include <numeric>

namespace framework {

Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    data_.resize(total_size, 0.0);
}

double& Tensor::operator[](size_t index) {
    return data_[index];
}

const double& Tensor::operator[](size_t index) const {
    return data_[index];
}


void Tensor::backward() {
    // Backpropagation logic
}

std::vector<double>& Tensor::get_data() {
    return data_;
}

const std::vector<double>& Tensor::get_data() const {
    return data_;
}

} // framework namespace