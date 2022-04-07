#include "ml_lib/model.h"

namespace ml_lib
{
    namespace layer_type
    {
        Linear::Linear(const unsigned int &input_dimensions, const unsigned int &output_dimensions, const Initializer &weights_initializer, const Initializer &bias_initializer) : m_weight_matrix(Tensor::Zeros({output_dimensions, input_dimensions}, true)),
                                                                                                                                                                                   m_bias_vector(Tensor::Zeros({output_dimensions, 1}, true))
        {
            weights_initializer(m_weight_matrix);
            bias_initializer(m_bias_vector);
        }
        Tensor Linear::FeedForward(const Tensor &input) const
        {
            unsigned int batchsize = input.get_shape()[1];
            return m_weight_matrix.MatrixMult(input) + m_bias_vector.Repeat(1, batchsize);
        }
        void Linear::LinkLearnableParameter(OptimizerBase *optimizer) {
            optimizer->Link(&m_weight_matrix);
            optimizer->Link(&m_bias_vector);
        }
        

        Softmax::Softmax(unsigned int axis) : m_axis(axis) {}
        Tensor Softmax::FeedForward(const Tensor &input) const
        {
            Tensor exp_input = Tensor::ElementwiseExp(input);
            Tensor exp_sum = exp_input;

            for (unsigned int i = 0; i <= m_axis; i++)
            {
                exp_sum = exp_sum.Sum(i);
            }

            for (unsigned int i = 0; i <= m_axis; i++)
            {
                exp_sum = exp_sum.Repeat(i, input.get_shape()[i]);
            }

            return exp_input.HadamardMult(exp_sum.ElementwisePow(Tensor::Scalar(-1.)));
        }
    
        Tensor Sigmoid::FeedForward(const Tensor& input) const {
            Tensor ones_tensor = Tensor::Ones(input.get_shape());
            return (ones_tensor + Tensor::ElementwiseExp(input.ScalarMult(Tensor::Scalar(-1.)))).ElementwisePow(Tensor::Scalar(-1.));
        }
    } // namespace layer_types
} // namespace ml_model