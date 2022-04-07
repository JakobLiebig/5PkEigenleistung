#ifndef ML_MODEL_HEADER_GUARD
#define ML_MODEL_HEADER_GUARD

#include <random>
#include <array>
#include <unordered_set>

#include "tensor.h"

namespace ml_lib
{
    typedef std::function<Tensor(const Tensor &x, const Tensor &target)> Lossfunction;
    namespace lossfunction
    {
        Tensor MeanSquaredError(const Tensor &x, const Tensor &target);
        Tensor CrossEntropy(const Tensor &x, const Tensor &target);
    } // namespace lossfunction

    typedef std::function<void(Tensor &learnable_parameter)> Initializer;
    namespace initializer
    {
        void Jakob(Tensor &learnable_parameter);
        void Zeros(Tensor &learnable_parameter);
    } // namespace initializer


    class LayerBase;
    class OptimizerBase;

    class LayerBase
    {
    public:
        LayerBase() = default;
        virtual ~LayerBase() = default;

        virtual Tensor FeedForward(const Tensor &input) const = 0;
        virtual void LinkLearnableParameter(OptimizerBase *optimizer) { };
    };
    namespace layer_type
    {
        class Linear : public LayerBase
        {
        public:
            Linear(const unsigned int& input_dimensions, const unsigned int& output_dimensions, const Initializer& weights_initializer, const Initializer& bias_initializer);
            ~Linear() override = default;

            Tensor FeedForward(const Tensor &input) const override;
            void LinkLearnableParameter(OptimizerBase *optimizer) override;
        
        private:
            Tensor m_weight_matrix;
            Tensor m_bias_vector;
        };
        class Conv2d : public LayerBase
        {
        };

        class Softmax : public LayerBase
        {
        public:
            Softmax(unsigned int axis);
            ~Softmax() override = default;

            Tensor FeedForward(const Tensor &input) const override;
            void LinkLearnableParameter(OptimizerBase *optimizer) override { };

        private:
            unsigned int m_axis;
        };
        class Batchnorm : public LayerBase
        {
        };

        class Sigmoid : public LayerBase
        {
        public:
            Sigmoid() = default;
            ~Sigmoid() override = default;

            Tensor FeedForward(const Tensor& input) const override;
            void LinkLearnableParameter(OptimizerBase *optimizer) override { };
        };
        class Relu : public LayerBase
        {
        };
    } // namespace layer_type

    class OptimizerBase
    {
    public:
        OptimizerBase() = default;
        virtual ~OptimizerBase() = default;

        virtual void Step(Tensor loss) = 0;
        virtual void Link(Tensor* learnable_parameter) = 0;
    };
    namespace optimizer
    {
        class Sgd : public OptimizerBase
        {
        };
        class MiniBatchSgd : public OptimizerBase
        {
        public:
            MiniBatchSgd(const std::vector<LayerBase *> &model_layers, const double &learning_rate);

            virtual void Step(Tensor loss) override;
            virtual void Link(Tensor* learnable_parameter) override;
        private:
            Tensor m_learning_rate;
            std::vector<Tensor *> m_learnable_parameters;
        };

        class Adam : public OptimizerBase
        {
        };
        class Adagrad : public OptimizerBase
        {
        };
        class Rmsprop : public OptimizerBase
        {
        };
    } // namespace optimizer
} // namespace ml_lib

#endif // !ML_MODEL_HEADER_GUARD