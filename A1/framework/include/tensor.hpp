#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>

namespace framework {

class Tensor {
public:
    Tensor(const std::vector<size_t>& shape);

    // Flat index access
    double& operator[](size_t index);
    const double& operator[](size_t index) const;

    // Gradient tracking
    void backward();

    // Data access
    std::vector<double>& get_data();
    const std::vector<double>& get_data() const;

private:
    std::vector<size_t> shape_;
    std::vector<double> data_;
    
};

}

#endif