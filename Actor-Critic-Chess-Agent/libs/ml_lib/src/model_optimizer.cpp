#include "ml_lib/model.h"

namespace ml_lib
{
    namespace optimizer
    {
        MiniBatchSgd::MiniBatchSgd(const std::vector<LayerBase *> &model_layers, const double &learning_rate) : m_learning_rate(Tensor::Scalar(learning_rate)),
                                                                                                                m_learnable_parameters()
        {
            for (LayerBase *layer : model_layers)
            {
                layer->LinkLearnableParameter(this);
            }
        }

        void MiniBatchSgd::Step(Tensor loss)
        {
            for(unsigned int i = 0; i < loss.get_dimensions(); i++)
                loss = loss.Sum(i);
            loss.Backward();

            for (Tensor *learnable_parameter : m_learnable_parameters)
            {
                unsigned int last_index = learnable_parameter->get_dimensions() - 1;
                int batchsize = learnable_parameter->get_shape()[last_index];

                Tensor derivative = learnable_parameter->Grad();
                derivative = derivative.ScalarMult(Tensor::Scalar(1. / (double)batchsize));
                Tensor new_l_p = *learnable_parameter - derivative.ScalarMult(m_learning_rate);

                learnable_parameter->SetElementValues(new_l_p);
            }
        }
        void MiniBatchSgd::Link(Tensor *learnable_parameter)
        {
            m_learnable_parameters.push_back(learnable_parameter);
        }
    } // namespace optimizer
} // namespace ml_lib