#include "ml_lib/tensor.h"

namespace ml_lib
{
	Tensor Tensor::Empty() 
	{
		return Tensor({});
	}

	Tensor Tensor::Ones(const std::vector<unsigned int> &shape, const bool &requires_grad)
	{
		Tensor ones(shape);

		for (unsigned int i = 0; i < ones.m_num_elements; i++)
		{
			ones.m_elements_ptr[i] = Element(1., requires_grad);
		}

		return ones;
	}
	Tensor Tensor::Zeros(const std::vector<unsigned int> &shape, const bool &requires_grad)
	{
		Tensor zeros(shape);

		for (unsigned int i = 0; i < zeros.m_num_elements; i++)
		{
			zeros.m_elements_ptr[i] = Element(0., requires_grad);
		}

		return zeros;
	}

	Tensor Tensor::Scalar(const double &value, const bool &requires_grad)
	{
		return Tensor({1}, &value, requires_grad);
	}

	Tensor::Tensor(const std::vector<unsigned int> &shape, const double *init_values, const bool &requires_grad) : m_num_elements(shape.size() != 0 ? 1 : 0),
																												   m_elements_ptr(nullptr),

																												   m_shape(shape)
	{
		for (unsigned int i = 0U; i < m_shape.size(); i++)
		{
			m_num_elements *= shape[i];
		}

		m_elements_ptr = new Element[m_num_elements];
		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			m_elements_ptr[i] = Element(init_values[i], requires_grad);
		}
	}
	Tensor::Tensor(const Tensor &obj) : m_num_elements(obj.m_num_elements),
										m_elements_ptr(new Element[m_num_elements]),
										m_shape(obj.m_shape)
	{
		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			m_elements_ptr[i] = obj.m_elements_ptr[i];
		}
	}
	Tensor::Tensor(Tensor &&obj) noexcept : m_num_elements(obj.m_num_elements),
											m_elements_ptr(obj.m_elements_ptr),
											m_shape(obj.m_shape)
	{
		obj.m_num_elements = 0;
		obj.m_elements_ptr = nullptr;
	}
	Tensor::~Tensor()
	{
		delete[] m_elements_ptr;
	}

	void Tensor::SetSingleElementValue(const double &new_value, const unsigned int &element_index)
	{
		if (element_index > m_num_elements)
			throw std::invalid_argument("index out of bounds!");

		m_elements_ptr[element_index].set_value(new_value);
	}
	void Tensor::SetElementValues(const Tensor &new_values)
	{
		if(new_values.m_num_elements != m_num_elements)
			throw;

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			m_elements_ptr[i].set_value(new_values.m_elements_ptr[i].get_value());
		}
	}
	void Tensor::SetElementValues(const double *new_values)
	{
		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			m_elements_ptr[i].set_value(new_values[i]);
		}
	}


    double Tensor::get_element_value_at(const int& index) const
	{
		return m_elements_ptr[index].get_value();
	}
	unsigned int Tensor::get_dimensions() const
	{
		return m_shape.size();
	}
	std::vector<unsigned int> Tensor::get_shape() const
	{
		return m_shape;
	}
	unsigned int Tensor::get_num_elements() const
	{
		return m_num_elements;
	}
	void Tensor::LogElementValues() const
	{
		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			std::cout << i << ". " << m_elements_ptr[i].get_value() << std::endl;
		}
	}

	Tensor Tensor::Grad() const
	{
		Tensor gradient_tensor = Tensor(m_shape);

		for(unsigned int i = 0; i < m_num_elements; i++)
			gradient_tensor.m_elements_ptr[i] = Element(m_elements_ptr[i].get_gradient());

		return gradient_tensor;
	}
	void Tensor::Backward()
	{
		if(m_num_elements != 1)
			throw;
		
		m_elements_ptr[0].Backward();
	}

	Tensor &Tensor::operator=(const Tensor &other)
	{
		if (&other != this)
		{
			m_num_elements = other.m_num_elements;
			
			delete[] m_elements_ptr;
			m_elements_ptr = new Element[m_num_elements];
			for (unsigned int i = 0; i < m_num_elements; i++)
			{
				m_elements_ptr[i] = other.m_elements_ptr[i];
			}

			m_shape = other.m_shape;
		}
		return *this;
	}
	Tensor &Tensor::operator=(Tensor &&other) noexcept
	{
		m_num_elements = other.m_num_elements;
		
		delete[] m_elements_ptr;
		m_elements_ptr = other.m_elements_ptr;
		
		m_shape = other.m_shape;

		other.m_num_elements = 0;
		other.m_elements_ptr = nullptr;
		other.m_shape.clear();

		return *this;
	}
	Tensor Tensor::operator+(const Tensor &other) const
	{
		if (m_num_elements != other.m_num_elements)
			throw;

		Tensor sum(m_shape);

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			sum.m_elements_ptr[i] = m_elements_ptr[i] + other.m_elements_ptr[i];
		}

		return sum;
	}
	Tensor Tensor::operator-(const Tensor &other) const
	{
		if (m_num_elements != other.m_num_elements)
			throw;

		Tensor difference(m_shape);

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			difference.m_elements_ptr[i] = m_elements_ptr[i] - other.m_elements_ptr[i];
		}

		return difference;
	}
	Tensor Tensor::operator-() const
	{
		Tensor negative(m_shape);

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			negative.m_elements_ptr[i] = -m_elements_ptr[i];
		}

		return negative;
	}

	Tensor Tensor::ElementwiseExp(const Tensor &exponent)
	{
		Tensor exp(exponent.m_shape);

		for (unsigned int i = 0; i < exponent.m_num_elements; i++)
		{
			exp.m_elements_ptr[i] = Element::exp(exponent.m_elements_ptr[i]);
		}

		return exp;
	}
	Tensor Tensor::ElementwisePow(const Tensor &scalar_exponent) const
	{
		Tensor pow(m_shape);

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			pow.m_elements_ptr[i] = m_elements_ptr[i].pow(*scalar_exponent.m_elements_ptr);
		}

		return pow;
	}
	Tensor Tensor::ElementwiseLog(const Tensor &scalar_base) const
	{
		Tensor log(m_shape);

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			log.m_elements_ptr[i] = m_elements_ptr[i].log(m_elements_ptr[i], *scalar_base.m_elements_ptr);
		}

		return log;
	}
	Tensor Tensor::ElementwiseMax(const Tensor &a, const Tensor &b)
	{
		Tensor max(a.m_shape);

		for (unsigned int i = 0; i < a.m_num_elements; i++)
		{
			Tensor::Element &a_element = a.m_elements_ptr[i];
			Tensor::Element &b_element = a.m_elements_ptr[i];

			if (a_element.get_value() >= b_element.get_value())
			{
				max.m_elements_ptr[i] = a_element;
			}
			else
			{
				max.m_elements_ptr[i] = b_element;
			}
		}

		return max;
	}

	Tensor Tensor::HadamardMult(const Tensor &other) const
	{
		if(other.m_num_elements != m_num_elements)
			throw;

		Tensor product(m_shape);

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			product.m_elements_ptr[i] = m_elements_ptr[i] * other.m_elements_ptr[i];
		}

		return product;
	}
	Tensor Tensor::ScalarMult(const Tensor &scalar) const
	{
		if(scalar.m_num_elements != 1)
			throw;

		Tensor product(m_shape);

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			product.m_elements_ptr[i] = *scalar.m_elements_ptr * m_elements_ptr[i];
		}

		return product;
	}
	Tensor Tensor::MatrixMult(const Tensor &other) const
	{
		// multiplier_columns = multiplicand_rows
		const unsigned int &multiplier_rows = m_shape[0];
		const unsigned int &multiplier_columns = m_shape[1];
		const unsigned int &multiplicand_rows = other.m_shape[0];
		const unsigned int &multiplicand_columns = other.m_shape[1];

		if(multiplier_columns != multiplicand_rows)
			throw;

		Tensor product({multiplier_rows, multiplicand_columns});

		for (unsigned int i = 0; i < multiplier_rows * multiplicand_columns; i++)
		{
			unsigned int x = i % multiplier_rows;
			unsigned int y = i / multiplier_rows;

			for (unsigned int j = 0; j < multiplier_columns; j++)
			{
				unsigned int multiplier_pos = x + j * multiplier_rows;
				unsigned int multiplicand_pos = j + y * multiplicand_rows;

				product.m_elements_ptr[i] = product.m_elements_ptr[i] + m_elements_ptr[multiplier_pos] * other.m_elements_ptr[multiplicand_pos];
			}
		}

		return product;
	}

	Tensor Tensor::Sum(const unsigned int &axis) const
	{
		if(axis >= m_shape.size() || axis < 0)
			throw;

		std::vector<unsigned int> sum_shape = m_shape;
		sum_shape[axis] = 1U;
		Tensor sum(sum_shape);

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			unsigned int z = i;
			unsigned dimension;
			unsigned int targetpos = 0U;
			unsigned int dimension_elementsize = 1U;

			for (unsigned int j = 0; j < m_shape.size(); j++)
			{
				dimension = z % m_shape[j];
				if (j != axis)
					targetpos += dimension * dimension_elementsize;

				dimension_elementsize *= sum_shape[j];
				z = (z - dimension) / m_shape[j];
			}

			sum.m_elements_ptr[targetpos] = sum.m_elements_ptr[targetpos] + m_elements_ptr[i];
		}

		return sum;
	}
	Tensor Tensor::Repeat(const unsigned int &axis, const unsigned int &repetitions) const
	{
		if(axis >= m_shape.size() || axis < 0)
			throw;

		std::vector<unsigned int> repeat_shape = m_shape;
		repeat_shape[axis] *= repetitions;
		Tensor repeat(repeat_shape);

		for (unsigned int i = 0; i < repeat.m_num_elements; i++)
		{
			unsigned int z = i;
			unsigned int dimension;
			unsigned int targetpos = 0;
			unsigned int dimension_elementsize = 1U;

			for (unsigned int j = 0; j < m_shape.size(); j++)
			{
				dimension = z % repeat_shape[j];

				unsigned int m = dimension;
				if (j == axis)
					m = m % m_shape[j];
				targetpos += m * dimension_elementsize;

				dimension_elementsize *= m_shape[j];
				z = (z - dimension) / repeat_shape[j];
			}

			repeat.m_elements_ptr[i] = m_elements_ptr[targetpos];
		}
		return repeat;
	}
	Tensor Tensor::Reshape(const std::vector<unsigned int> &new_shape) const
	{
		Tensor reshaped_tensor(*this);

		reshaped_tensor.m_shape = new_shape;
		return reshaped_tensor;
	}
	Tensor Tensor::Concatenate(const Tensor &a, const Tensor &b, unsigned int axis)
	{
		std::vector<unsigned int> concat_shape = a.m_shape;
		concat_shape[axis] += b.m_shape[axis];
		Tensor concat(concat_shape);
		
		int a_subtensor_size = 1;
		int b_subtensor_size = 1;

		for(unsigned int i = 0; i <= axis; i++) {
			a_subtensor_size *= a.m_shape[i];
			b_subtensor_size *= b.m_shape[i];
		}

		int pos_a = 0;
		int pos_b = 0;
		int pos_concat = 0;

		while (pos_concat < concat.m_num_elements) {
			for(unsigned int j = 0; j < a_subtensor_size; j++) {
				concat.m_elements_ptr[pos_concat] = a.m_elements_ptr[pos_a];

				pos_a++;
				pos_concat++;
			}

			for(unsigned int j = 0; j < b_subtensor_size; j++) {
				concat.m_elements_ptr[pos_concat] = b.m_elements_ptr[pos_b];
				
				pos_b++;
				pos_concat++;
			}
		}

		return concat;
	}

	unsigned int Tensor::ArgFind(const std::function<bool(const double &)> &find_func) const
	{
		unsigned int index;

		for (unsigned int i = 0; i < m_num_elements; i++)
		{
			if (find_func(m_elements_ptr[i].get_value()))
			{
				index = i;
			}
		}

		return index;
	}

	unsigned int Tensor::PositionToIndex(const std::vector<unsigned int> &position) const
	{
		unsigned int index = 0;
		
		for (int i = position.size() - 1; i >= 0; i--)
		{
			index = position[i] + m_shape[i] * index;
		}

		return index;
	}
	std::vector<unsigned int> Tensor::IndexToPosition(const unsigned int &index) const
	{
		unsigned int shape_dimensions = m_shape.size();
		std::vector<unsigned int> position(shape_dimensions);

		unsigned int z = index;
		for (unsigned int i = 0; i < shape_dimensions - 1; i++)
		{
			unsigned int pos = z % m_shape[i];
			z = (z - pos) / m_shape[i];

			position[i] = pos;
		}

		return position;
	}

	Tensor::Tensor(const std::vector<unsigned int> &shape) : m_num_elements(shape.size() != 0 ? 1 : 0),
															 m_elements_ptr(nullptr),
															 m_shape(shape)
	{

		for (unsigned int i = 0U; i < m_shape.size(); i++)
		{
			m_num_elements *= shape[i];
		}

		m_elements_ptr = new Element[m_num_elements];
	}
} // namespace ml_lib