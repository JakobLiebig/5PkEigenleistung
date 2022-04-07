#include "actor-critic-chess-agent/environment.h"

#define LOG(x) std::cout << x << std::endl

namespace chess_agent{
    void test(std::vector<ml_lib::LayerBase*> actor_model) {
        // commandline plays white
        // agent plays black

        chess_agent::Environment env;

        while(true) {
            // commandline takes action
            LOG("Commandline takes action!");
            LOG(env.get_board().ToString());

            std::string in_from, in_to;
            
            std::cout << "from>";
            std::cin >> in_from;
            
            std::cout << "to>";
            std::cin >> in_to;

            chess::Move commandline_action(in_from, in_to);

            if(env.MovePiece(commandline_action))
                exit;

            // agent takes action
            LOG("Agent takes action!");
            LOG(env.get_board().ToString());

            auto board_state = env.GenerateBoardState();
            auto action_space = env.GenerateActionSpace();

            if(env.get_active_player() != env.cDefaultViewPoint)
                board_state = env.SwitchBoardStatePov(board_state);

            auto agent_action_prop_distr = chess_agent::ActorFeedForward(board_state, action_space, actor_model);
            chess::Move agent_action = env.ActionPropDistrToMove(agent_action_prop_distr);

            if(env.MovePiece(agent_action))
                exit;
        }

        LOG("Game Over!");

        auto winner = env.get_active_player();
        if(winner == chess::Piece::Colour::White)
            LOG("Fuck you, i'll get you next time!");
        else
            LOG("Bitch get rekt!");
    }
} // namespace chess_agent