#ifndef ML_TENSOR_HEADER_GUARD
#define ML_TENSOR_HEADER_GUARD

#include <memory> // for std::shared_ptr
#include <vector> // for std::vector
#include <iostream> // for std::cout
#include <cmath> // for ...
#include <cstring>
#include <functional> // for std::function
#include <deque> // for std::deque

#define LOG(x) std::cout << x << std::endl

namespace ml_lib
{
    class Tensor
    {
    public:
        static Tensor Empty();
        static Tensor Ones(const std::vector<unsigned int> &shape, const bool &requires_grad = false);
        static Tensor Zeros(const std::vector<unsigned int> &shape, const bool &requires_grad = false);
        
        static Tensor Scalar(const double &value, const bool &requires_grad = false);

        Tensor(const std::vector<unsigned int> &shape, const double *init_values, const bool &requires_grad = false);
        Tensor(const Tensor &other);
        Tensor(Tensor &&obj) noexcept;
        ~Tensor();

        void SetSingleElementValue(const double& new_value, const unsigned int& element_index);
        void SetElementValues(const Tensor &new_values);
        void SetElementValues(const double *new_values);

        double get_element_value_at(const int& index) const;
        unsigned int get_dimensions() const;
        std::vector<unsigned int> get_shape() const;
        unsigned int get_num_elements() const;
        void LogElementValues() const;

        Tensor Grad() const;
        void Backward();

        Tensor &operator=(const Tensor &other);
        Tensor &operator=(Tensor &&other) noexcept;
        Tensor operator+(const Tensor &other) const;
        Tensor operator-(const Tensor &other) const;
        Tensor operator-() const;

        static Tensor ElementwiseExp(const Tensor &exponent);
        Tensor ElementwisePow(const Tensor &scalar_exponent) const;
        Tensor ElementwiseLog(const Tensor &scalar_base) const;
        static Tensor ElementwiseMax(const Tensor& a, const Tensor& b);

        Tensor HadamardMult(const Tensor &other) const;
        Tensor ScalarMult(const Tensor &scalar) const;
        Tensor MatrixMult(const Tensor &other) const;
        
        Tensor Conv2d() const;
        
        Tensor Sum(const unsigned int &axis) const;
        Tensor Repeat(const unsigned int &axis, const unsigned int &repetitions) const;
        Tensor Reshape(const std::vector<unsigned int>& shape) const;
        static Tensor Concatenate(const Tensor& a, const Tensor& b, unsigned int axis);
        unsigned int ArgFind(const std::function< bool(const double&)>& find_func) const;
        
        unsigned int PositionToIndex(const std::vector<unsigned int>& position) const;
        std::vector<unsigned int> IndexToPosition(const unsigned int& index) const;

    private:
        class Element
        {
        public:
            Element(const double &value = 0, const bool &requires_grad = false);
            Element(const Element &obj);
            ~Element() = default;

            void set_value(const double &new_value);
            double get_value() const;
            double get_gradient() const;

            void Backward();

            Element &operator=(const Element &other);
            Element &operator=(Element &&other) noexcept;

            Element operator*(const Element &other) const;
            Element operator/(const Element &other) const;
            Element operator+(const Element &other) const;
            Element operator-(const Element &other) const;
            Element pow(const Element &other) const;

            Element operator-() const;
            static Element exp(const Element &x);
            static Element ln(const Element &x);
            static Element log(const Element &x, const Element &base);
            static Element sin(const Element &x);
            static Element cos(const Element &x);
        private:
            class AutodiffNodeBase
            {
            public:
                AutodiffNodeBase();
                virtual ~AutodiffNodeBase() = default;

                virtual void Backward(std::deque<AutodiffNodeBase *> &queue) = 0;

                bool TryInitialize(const double& init_gradient);
                void Reset();
            
                bool allChildrenVisited();

                void AddChild();
                void RemoveChild();

                double m_gradient;
            
            private:
                unsigned int m_num_children;
                int m_unvisited_children;
            };
            class AutodiffRootNode : public AutodiffNodeBase
            {
            public:
                AutodiffRootNode(const std::shared_ptr<AutodiffNodeBase>& parent_a_ptr, const std::shared_ptr<AutodiffNodeBase>& parent_b_ptr, const double &parent_a_partial_derivative, const double &parent_b_partial_derivative);
                AutodiffRootNode(const AutodiffRootNode&) = delete;
                ~AutodiffRootNode() override;

                void Backward(std::deque<AutodiffNodeBase *> &queue) override;

            private:
                std::shared_ptr<AutodiffNodeBase> m_parent_a_ptr, m_parent_b_ptr;
                double m_parent_a_partial_derivative, m_parent_b_partial_derivative;
            };
            class AutodiffLeafNode : public AutodiffNodeBase
            {
            public:
                AutodiffLeafNode() = default;
                AutodiffLeafNode(const AutodiffLeafNode&) = delete;
                ~AutodiffLeafNode() override = default;

                void Backward(std::deque<AutodiffNodeBase *>&) override { };
            };

            Element(const double &value, const bool &requires_grad, std::shared_ptr<AutodiffNodeBase> &node_dependency);

            double m_value;

            bool m_requires_grad;
            std::shared_ptr<AutodiffNodeBase> m_node_dependency;
        };

        Tensor(const std::vector<unsigned int>& shape);

        unsigned int m_num_elements;
        Element *m_elements_ptr;

        std::vector<unsigned int> m_shape;
    };
} // namespace ml_lib

#endif // !ML_TENSOR_HEADER_GUARD