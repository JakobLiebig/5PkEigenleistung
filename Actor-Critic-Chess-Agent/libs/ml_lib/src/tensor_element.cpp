#include "ml_lib/tensor.h"

namespace ml_lib
{
	Tensor::Element::Element(const double &x, const bool &requires_grad) : m_value(x),
																		   m_requires_grad(requires_grad),
																		   m_node_dependency(nullptr)
	{
		if (requires_grad)
			m_node_dependency.reset(new AutodiffLeafNode());
	}
	Tensor::Element::Element(const double &x, const bool &requires_grad, std::shared_ptr<AutodiffNodeBase> &node_dependency) : m_value(x),
																															   m_requires_grad(requires_grad),
																															   m_node_dependency(node_dependency)
	{
	}
	Tensor::Element::Element(const Element &obj) : m_value(obj.m_value),
												   m_requires_grad(obj.m_requires_grad),
												   m_node_dependency(obj.m_node_dependency)
	{
	}

	void Tensor::Element::set_value(const double &new_value)
	{
		m_value = new_value;
	}
	double Tensor::Element::get_value() const
	{
		return m_value;
	}
	double Tensor::Element::get_gradient() const
	{
		if(!m_requires_grad)
			throw;
		
		return m_node_dependency.get()->m_gradient;
	}

	void Tensor::Element::Backward()
	{
		if(!m_requires_grad)
			throw;

		std::deque<AutodiffNodeBase *> queue;

		// initialize queue
		AutodiffNodeBase *init_node = m_node_dependency.get();

		init_node->TryInitialize(1.);

		queue.push_back(init_node);

		while (queue.size() > 0)
		{
			queue[0]->Backward(queue);
			queue.pop_front();
		}
	}

	Tensor::Element &Tensor::Element::operator=(const Element &other)
	{
		m_value = other.m_value;
		m_requires_grad = other.m_requires_grad;

		if (other.m_requires_grad)
		{
			m_node_dependency.reset(new AutodiffRootNode(other.m_node_dependency, nullptr, 1., 0.));
		}
		else
		{
			m_node_dependency = nullptr;
		}
		return *this;
	}
	Tensor::Element &Tensor::Element::operator=(Element &&other) noexcept
	{
		m_value = other.m_value;
		m_requires_grad = other.m_requires_grad;
		m_node_dependency = other.m_node_dependency;

		return *this;
	}

	Tensor::Element Tensor::Element::operator*(const Element &other) const
	{
		double product_x = m_value * other.m_value;
		bool product_requires_grad = m_requires_grad || other.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> product_node_dependency = nullptr;

		if (product_requires_grad)
		{
			double parent_a_partiall = other.m_value;
			double parent_b_partiall = m_value;

			product_node_dependency.reset(new AutodiffRootNode(m_node_dependency, other.m_node_dependency, parent_a_partiall, parent_b_partiall));
		}

		return Element(product_x, product_requires_grad, product_node_dependency);
	}
	Tensor::Element Tensor::Element::operator/(const Element &other) const
	{
		double fraction_x = m_value * other.m_value;
		bool fraction_requires_grad = m_requires_grad || other.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> fraction_node_dependency = nullptr;

		if (fraction_requires_grad)
		{
			double parent_a_partiall = 1. / other.m_value;
			double parent_b_partiall = -m_value / std::pow(other.m_value, 2.);

			fraction_node_dependency.reset(new AutodiffRootNode(m_node_dependency, other.m_node_dependency, parent_a_partiall, parent_b_partiall));
		}

		return Element(fraction_x, fraction_requires_grad, fraction_node_dependency);
	}
	Tensor::Element Tensor::Element::operator+(const Element &other) const
	{
		double sum_value = m_value + other.m_value;
		bool sum_requires_grad = m_requires_grad || other.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> sum_node_dependency = nullptr;

		if (sum_requires_grad)
		{
			double parent_a_partiall = 1.;
			double parent_b_partiall = 1.;

			sum_node_dependency.reset(new AutodiffRootNode(m_node_dependency, other.m_node_dependency, parent_a_partiall, parent_b_partiall));
		}

		return Element(sum_value, sum_requires_grad, sum_node_dependency);
	}
	Tensor::Element Tensor::Element::operator-(const Element &other) const
	{
		double difference_x = m_value - other.m_value;
		bool difference_requires_grad = m_requires_grad || other.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> difference_node_dependency = nullptr;

		if (difference_requires_grad)
		{
			double parent_a_partiall = 1.;
			double parent_b_partiall = -1.;

			difference_node_dependency.reset(new AutodiffRootNode(m_node_dependency, other.m_node_dependency, parent_a_partiall, parent_b_partiall));
		}

		return Element(difference_x, difference_requires_grad, difference_node_dependency);
	}
	Tensor::Element Tensor::Element::pow(const Element &other) const
	{
		double power_x = std::pow(m_value, other.m_value);
		bool power_requires_grad = m_requires_grad || other.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> power_node_dependency = nullptr;

		if (power_requires_grad)
		{
			double parent_a_partiall = other.m_value * std::pow(m_value, other.m_value - 1.);
			double parent_b_partiall = -1.;

			power_node_dependency.reset(new AutodiffRootNode(m_node_dependency, other.m_node_dependency, parent_a_partiall, parent_b_partiall));
		}

		return Element(power_x, power_requires_grad, power_node_dependency);
	}
	Tensor::Element Tensor::Element::log(const Element &other, const Element &base)
	{
		double log_x = std::log(other.m_value) / std::log(base.m_value);
		bool log_requires_grad = other.m_requires_grad || base.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> log_node_dependency = nullptr;

		if (log_requires_grad)
		{
			double parent_a_partiall = 1. / (other.m_value * std::log(base.m_value));
			double parent_b_partiall = std::log(other.m_value) / std::log(base.m_value);

			log_node_dependency.reset(new AutodiffRootNode(other.m_node_dependency, base.m_node_dependency, parent_a_partiall, parent_b_partiall));
		}

		return Element(log_x, log_requires_grad, log_node_dependency);
	}

	Tensor::Element Tensor::Element::operator-() const
	{
		double difference_x = -1. * m_value;
		bool difference_requires_grad = m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> difference_node_dependency = nullptr;

		if (difference_requires_grad)
		{
			double parent_partiall = -1.;

			difference_node_dependency.reset(new AutodiffRootNode(m_node_dependency, nullptr, parent_partiall, 0.));
		}

		return Element(difference_x, difference_requires_grad, difference_node_dependency);
	}
	Tensor::Element Tensor::Element::exp(const Element &x)
	{
		double exponent_x = std::exp(x.m_value);
		bool exponent_requires_grad = x.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> exponent_node_dependency = nullptr;

		if (exponent_requires_grad)
		{
			double parent_partiall = std::exp(x.m_value);

			exponent_node_dependency.reset(new AutodiffRootNode(x.m_node_dependency, nullptr, parent_partiall, 0.));
		}

		return Element(exponent_x, exponent_requires_grad, exponent_node_dependency);
	}
	Tensor::Element Tensor::Element::ln(const Element &x)
	{
		double log_x = std::log(x.m_value);
		bool log_requires_grad = x.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> log_node_dependency = nullptr;

		if (log_requires_grad)
		{
			double parent_partiall = 1. / x.m_value;

			log_node_dependency.reset(new AutodiffRootNode(x.m_node_dependency, nullptr, parent_partiall, 0.));
		}

		return Element(log_x, log_requires_grad, log_node_dependency);
	}
	Tensor::Element Tensor::Element::sin(const Element &x)
	{
		double sin_x = std::sin(x.m_value);
		bool sin_requires_grad = x.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> sin_node_dependency = nullptr;

		if (sin_requires_grad)
		{
			double parent_partiall = std::cos(x.m_value);

			sin_node_dependency.reset(new AutodiffRootNode(x.m_node_dependency, nullptr, parent_partiall, 0.));
		}

		return Element(sin_x, sin_requires_grad, sin_node_dependency);
	}
	Tensor::Element Tensor::Element::cos(const Element &x)
	{
		double cos_x = std::cos(x.m_value);
		bool cos_requires_grad = x.m_requires_grad;
		std::shared_ptr<AutodiffNodeBase> cos_node_dependency = nullptr;

		if (cos_requires_grad)
		{
			double parent_partiall = -1. * std::sin(x.m_value);

			cos_node_dependency.reset(new AutodiffRootNode(x.m_node_dependency, nullptr, parent_partiall, 0.));
		}

		return Element(cos_x, cos_requires_grad, cos_node_dependency);
	}
} // namespace ml_lib