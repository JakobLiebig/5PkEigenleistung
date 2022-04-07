#include "actor-critic-chess-agent/environment.h"

namespace chess_agent
{
    Replay::Replay() : m_state(ml_lib::Tensor::Empty()),
                       m_next_state(ml_lib::Tensor::Empty()),
                       m_action(ml_lib::Tensor::Empty()),
                       m_action_space(ml_lib::Tensor::Empty()),
                       m_return(ml_lib::Tensor::Empty())
    {

    }

    Replay::Replay(const ml_lib::Tensor &state,
                   const ml_lib::Tensor &next_state,
                   const ml_lib::Tensor &action_prop_distr,
                   const ml_lib::Tensor &action_space,
                   const ml_lib::Tensor &return_) : m_state(state),
                                            m_next_state(next_state),
                                            m_action(action_prop_distr),
                                            m_action_space(action_space),
                                            m_return(return_)
    {
    }

    Replay Replay::Concatenate(const Replay &a, const Replay &b)
    {
        // shape of chess state: 8, 8, 16, 2, 1
        ml_lib::Tensor state = ml_lib::Tensor::Concatenate(a.m_state, b.m_state, 4);
        ml_lib::Tensor next_state = ml_lib::Tensor::Concatenate(a.m_next_state, b.m_next_state, 4);

        // shape of action: 8, 8, 16, 1
        ml_lib::Tensor action = ml_lib::Tensor::Concatenate(a.m_action, b.m_action, 3);
        ml_lib::Tensor action_space = ml_lib::Tensor::Concatenate(a.m_action_space, b.m_action_space, 3);

        // shape of return: 1
        ml_lib::Tensor return_ = ml_lib::Tensor::Concatenate(a.m_return, b.m_return, 0);

        return Replay(state,
                      next_state,
                      action,
                      action_space,
                      return_);
    }

    ml_lib::Tensor Replay::get_state() const
    {
        return m_state;
    }
    ml_lib::Tensor Replay::get_next_state() const
    {
        return m_next_state;
    }

    ml_lib::Tensor Replay::get_action() const
    {
        return m_action;
    }
    ml_lib::Tensor Replay::get_action_space() const
    {
        return m_action_space;
    }

    ml_lib::Tensor Replay::get_return() const
    {
        return m_return;
    }
    void Replay::set_return(const ml_lib::Tensor &new_return)
    {
        m_return = new_return;
    }

} // namespace chess_agent