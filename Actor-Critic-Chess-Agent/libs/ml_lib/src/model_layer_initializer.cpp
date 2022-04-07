#include "ml_lib/model.h"

namespace ml_lib
{
    namespace initializer {
        void Jakob(Tensor &learnable_parameter) {
            unsigned int l_p_num_of_elements = learnable_parameter.get_num_elements();
            unsigned int l_p_num_of_elements_double = (double)l_p_num_of_elements;

            double* l_p_contents = new double[l_p_num_of_elements];

            static double jakob_constant = 4. / (double)RAND_MAX;

            for(unsigned int i = 0; i < l_p_num_of_elements; i++) {
                l_p_contents[i] = ((double)rand() / l_p_num_of_elements_double) * jakob_constant; 
            }

            learnable_parameter.SetElementValues(l_p_contents);
            delete[] l_p_contents;
        }
    } // namespace initializer
} // namespace ml_lib