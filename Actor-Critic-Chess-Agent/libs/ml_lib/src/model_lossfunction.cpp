#include "ml_lib/model.h"

namespace ml_lib {
    namespace lossfunction {
        Tensor MeanSquaredError(const Tensor &x, const Tensor &target) {
            Tensor loss = (x - target).ElementwisePow(Tensor::Scalar(2.));

            int batchsize = x.get_shape()[x.get_dimensions() -1];
            int num_elements_per_batch = x.get_num_elements() / batchsize;

            for(unsigned int i = 0; i < loss.get_dimensions()-1; i++){
                loss = loss.Sum(i);
            }
            
            return loss.ScalarMult(Tensor::Scalar(1. / (double)num_elements_per_batch));
        }

        Tensor CrossEntropy(const Tensor &x, const Tensor &target) {
            Tensor loss = target.HadamardMult(x.ElementwiseLog(Tensor::Scalar(2)));

            unsigned int batch_element_dimensions = loss.get_dimensions()-2;
            for(unsigned int i = 0; i < 1; i++) // 
            {
                loss = loss.Sum(i);
            }

            return loss.ScalarMult(Tensor::Scalar(-1.));
        }
    } // namespace lossfunction
} // namespace ml_lib