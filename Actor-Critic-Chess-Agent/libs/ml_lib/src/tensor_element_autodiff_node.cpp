#include "ml_lib/tensor.h"

namespace ml_lib
{
    Tensor::Element::AutodiffNodeBase::AutodiffNodeBase() : m_gradient(0.),
                                                m_num_children(0U),
                                                m_unvisited_children(-1)
    {
    }

    bool Tensor::Element::AutodiffNodeBase::TryInitialize(const double& init_gradient)
    {
        if (m_unvisited_children == -1)
        {
            m_unvisited_children = m_num_children;
            m_gradient = init_gradient;

            return true;
        }

        return false;
    }
    void Tensor::Element::AutodiffNodeBase::Reset()
    {
        m_unvisited_children = -1;
    }

    bool Tensor::Element::AutodiffNodeBase::allChildrenVisited()
    {
        // allChildrenVisited gets called when a children is visited
        // thus the counter decreases by one 
        m_unvisited_children -= 1;

        return m_unvisited_children == 0;
    }

    void Tensor::Element::AutodiffNodeBase::AddChild() {
        m_num_children += 1U;
    }
    void Tensor::Element::AutodiffNodeBase::RemoveChild() {
        m_num_children -= 1U;
    }

    Tensor::Element::AutodiffRootNode::AutodiffRootNode(const std::shared_ptr<AutodiffNodeBase>& parent_a_ptr, const std::shared_ptr<AutodiffNodeBase>& parent_b_ptr, const double &parent_a_partial_derivative, const double &parent_b_partial_derivative) : m_parent_a_ptr(parent_a_ptr),
                                                                                                                                                                                                                        m_parent_b_ptr(parent_b_ptr),
                                                                                                                                                                                                                        m_parent_a_partial_derivative(parent_a_partial_derivative),
                                                                                                                                                                                                                        m_parent_b_partial_derivative(parent_b_partial_derivative)
    {
        if (m_parent_a_ptr.get() != nullptr)
            m_parent_a_ptr.get()->AddChild();
        if (m_parent_b_ptr.get() != nullptr)
            m_parent_b_ptr.get()->AddChild();
    }
    Tensor::Element::AutodiffRootNode::~AutodiffRootNode()
    {
        if (m_parent_a_ptr.get() != nullptr)
            m_parent_a_ptr.get()->RemoveChild();
        if (m_parent_b_ptr.get() != nullptr)
            m_parent_b_ptr.get()->RemoveChild();
    }
    void Tensor::Element::AutodiffRootNode::Backward(std::deque<AutodiffNodeBase *> &queue)
    {
        if (m_parent_a_ptr.get() != nullptr)
        {
            m_parent_a_ptr.get()->TryInitialize(0.);
            
            m_parent_a_ptr.get()->m_gradient += m_gradient * m_parent_a_partial_derivative;

            if (m_parent_a_ptr.get()->allChildrenVisited())
                queue.push_back(m_parent_a_ptr.get());
        }

        if (m_parent_b_ptr.get() != nullptr)
        {
            m_parent_b_ptr.get()->TryInitialize(0.);
            
            m_parent_b_ptr.get()->m_gradient += m_gradient * m_parent_b_partial_derivative;

            if (m_parent_b_ptr.get()->allChildrenVisited())
                queue.push_back(m_parent_b_ptr.get());
        }

        this->Reset();
    }
} // namespace ml_tensor