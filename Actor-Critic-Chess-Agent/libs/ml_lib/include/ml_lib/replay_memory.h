#ifndef ML_REPLAY_MEMORY_HEADER_GUARD
#define ML_REPLAY_MEMORY_HEADER_GUARD

#include <vector>
#include <unordered_set>

namespace ml_lib
{
    template <typename T>
    class ReplayMemory
    {
    public:
        ReplayMemory(const int &max_elements) : m_cur_num_elements(0U),
                                                m_max_elements(max_elements),
                                                m_position(0U),
                                                m_state_action_pairs(new T[max_elements])
        {
        }
        ReplayMemory(const ReplayMemory &obj) = delete;
        ~ReplayMemory()
        {
            delete[] m_state_action_pairs;
        }

        void Put(const T &state_action_pair)
        {
            m_state_action_pairs[m_position] = state_action_pair;

            m_cur_num_elements = std::min((m_cur_num_elements + 1), m_max_elements);
            m_position = (m_position + 1) % m_max_elements;
        }

        template <int BATCH_SIZE>
        T GenerateRandomBatch()
        {
            if(BATCH_SIZE > (int)m_cur_num_elements)
                throw;

            std::array<int, BATCH_SIZE> random_batch_indexies = GenerateRandomBatchIndexies<BATCH_SIZE>(m_cur_num_elements);

            T batch = m_state_action_pairs[random_batch_indexies[0]];

            for (unsigned int i = 1; i < BATCH_SIZE; i++)
            {
                int random_index = random_batch_indexies[i];
                batch = T::Concatenate(batch, m_state_action_pairs[random_index]);
            }

            return batch;
        }

    private:
        template <int BATCH_SIZE>
        static std::array<int, BATCH_SIZE> GenerateRandomBatchIndexies(int max_index)
        {
            std::unordered_set<int> set_of_indexies;
            std::array<int, BATCH_SIZE> array_of_indexies;

            for (unsigned int i = 0; i < BATCH_SIZE;)
            {
                int random_index = std::rand() % max_index;

                if (set_of_indexies.find(random_index) == set_of_indexies.end())
                {
                    set_of_indexies.insert(random_index);
                    array_of_indexies[i] = random_index;

                    i++;
                }
            }

            return array_of_indexies;
        }

        unsigned int m_cur_num_elements;
        unsigned int m_max_elements;
        unsigned int m_position;

        T *m_state_action_pairs;
    };
} // namespace ml_lib

#endif // !ML_REPLAY_MEMORY_HEADER_GUARD