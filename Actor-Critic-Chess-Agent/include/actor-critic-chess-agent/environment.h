#ifndef CHESS_AGENT_ENVIRONMENT_HEADER_GUARD
#define CHESS_AGENT_ENVIRONMENT_HEADER_GUARD

#include <array>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "ml_lib/tensor.h"
#include "ml_lib/model.h"
#include "ml_lib/replay_memory.h"

#include "chess_lib/chess.h"

namespace chess_agent
{
    extern ml_lib::Tensor ActorFeedForward(const ml_lib::Tensor& board_state, const ml_lib::Tensor& action_space, std::vector<ml_lib::LayerBase*> actor_model);

    extern void train(const int& epochs, std::vector<ml_lib::LayerBase*>& actor_model);
    extern void test(std::vector<ml_lib::LayerBase*> actor_model);

    class Replay
    {
    public:
        Replay();
        Replay(const ml_lib::Tensor &state,
               const ml_lib::Tensor &next_state,
               const ml_lib::Tensor &action_prop_distr,
               const ml_lib::Tensor &action_space,
               const ml_lib::Tensor &return_);
        
        static Replay Concatenate(const Replay &a, const Replay &b);

        ml_lib::Tensor get_state() const;
        ml_lib::Tensor get_next_state() const;

        ml_lib::Tensor get_action() const;
        ml_lib::Tensor get_action_space() const;

        ml_lib::Tensor get_return() const;
        void set_return(const ml_lib::Tensor &new_return);

    private:
        ml_lib::Tensor m_state;
        ml_lib::Tensor m_next_state;

        ml_lib::Tensor m_action;
        ml_lib::Tensor m_action_space;

        ml_lib::Tensor m_return;
    };

    class Environment
    {
    public:
        Environment();

        static ml_lib::Tensor SwitchBoardStatePov(const ml_lib::Tensor &board_state);

        chess::Move ActionPropDistrToMove(const ml_lib::Tensor &action_prop_distr);
        ml_lib::Tensor MoveToActionPropDistr(const chess::Move &move);

        chess::Board get_board() const;
        std::vector<chess::Move> get_legal_moves() const;
        chess::Piece::Colour get_active_player() const;

        bool MovePiece(const chess::Move& move);
        void Reset();

        ml_lib::Tensor GenerateBoardState() const;
        ml_lib::Tensor GenerateActionSpace() const;

        static const chess::Piece::Colour cDefaultViewPoint;
        static const chess::Piece::Colour cFirstIdPieceColour;
    private:
        static std::array<int, 32> BasicPiecePositions();

        std::array<int, 32> m_piece_positions;
        chess::Game m_game;
    };

} // namespace chess_agent

#endif // !CHESS_AGENT_ENVIRONMENT_HEADER_GUARD