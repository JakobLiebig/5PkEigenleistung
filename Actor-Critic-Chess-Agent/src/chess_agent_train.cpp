#include "actor-critic-chess-agent/environment.h"

#define BATCHSIZE 1

const int cReplayMemorySize = 100;
const double cEpsilonDecayA = 0.5;
const double cEpsilonDecayB = 0.1;
const double cEpsylonDecayC = 0.1;

const int max_round_per_game = 50;

ml_lib::Tensor CriticFeedForward(const ml_lib::Tensor& board_state, const ml_lib::Tensor& action_prop_distr, std::vector<ml_lib::LayerBase*> critic_model) {
    // board_state shape: {8, 8, 16, 2, 1}
    // action_prop_distr shape {8, 8, 16, 1}
    unsigned int batchsize = action_prop_distr.get_num_elements() / 1024;
    auto action_prop_distr_ = action_prop_distr.Reshape({8, 8, 16, 1, batchsize});

    auto out = ml_lib::Tensor::Concatenate(board_state, action_prop_distr_, 3);

    out = out.Reshape({3072, batchsize});

    for(auto layer: critic_model) {
        out = layer->FeedForward(out);
    }

    return out;
}

ml_lib::Tensor GenerateReturn(const chess::Piece::Colour& winner, const chess::Piece::Colour& cur_player) {
    if(winner == cur_player)
        return ml_lib::Tensor::Scalar(1.);
    else
        return ml_lib::Tensor::Scalar(0.);
}

bool EpsylonGreedy(const int& epochs) {
    static int t = 0;
    t += 1;
    
    double standardized_time = (t-cEpsilonDecayA * (double)epochs) / (cEpsilonDecayB * (double)epochs);
    double cosh_ = cosh(exp(-standardized_time));
    double epsilon = 1.1 - ( 1. / cosh_ + (t * cEpsylonDecayC / (double)epochs));

    double rand_ = (double)rand() / (double)RAND_MAX;

    // returns true for actor taking action, false for random action
    return rand_ > epsilon;
}

namespace chess_agent {
    void train(const int& epochs, std::vector<ml_lib::LayerBase*>& actor_model)
    {
        ml_lib::layer_type::Linear critic_l1(3072, 1000, ml_lib::initializer::Jakob, ml_lib::initializer::Jakob);
        ml_lib::layer_type::Linear critic_l2(1000, 500, ml_lib::initializer::Jakob, ml_lib::initializer::Jakob);
        ml_lib::layer_type::Linear critic_l3(500, 1, ml_lib::initializer::Jakob, ml_lib::initializer::Jakob);
        std::vector<ml_lib::LayerBase*> critic_model = {&critic_l1, &critic_l2, &critic_l3};
        ml_lib::Lossfunction critic_lossfunc = ml_lib::lossfunction::CrossEntropy;
        
        ml_lib::optimizer::MiniBatchSgd actor_optimizer(actor_model, 0.1);
        ml_lib::optimizer::MiniBatchSgd critic_optimizer(critic_model, 0.1);


        chess_agent::Environment env;
        ml_lib::ReplayMemory<chess_agent::Replay> rm(cReplayMemorySize);

        for(unsigned int epoch_id = 0; epoch_id < epochs; epoch_id++) {
            std::cout << "[+] " << epoch_id << ". Round begins!" << std::endl;

            env.Reset();

            // play one game
            bool gameover = false;
            int game_index = 0;

            std::vector<chess_agent::Replay> game_replays;
            auto next_state = env.GenerateBoardState();

            while(!gameover && game_index <= max_round_per_game) {
                game_index++;

                auto cur_state = std::move(next_state);

                auto action_space = env.GenerateActionSpace();

                auto action_prop_distr = ml_lib::Tensor::Empty();
                chess::Move action;
                                
                if(EpsylonGreedy(epochs)) {
                    // actor takes action
                    action_prop_distr = ActorFeedForward(cur_state,
                                                        action_space,
                                                        actor_model);

                    action = env.ActionPropDistrToMove(action_prop_distr);
                } else {
                    // random action
                    auto legal_moves = env.get_legal_moves();
                                    
                    auto random_action_index = rand() % legal_moves.size();
                    action = legal_moves[random_action_index];

                    action_prop_distr = env.MoveToActionPropDistr(action);
                }

                gameover = env.MovePiece(action);

                next_state = env.GenerateBoardState();

                if(env.get_active_player() != env.cDefaultViewPoint) {
                    // The agent is suppose to play against itself. That means he takes actions from pov.black and pov.white.
                    // Normalize cur_state and next_state so both are from pov=active_player
                    // (meaning he always views the board from his point of view)
                    
                    cur_state = chess_agent::Environment::SwitchBoardStatePov(cur_state);
                    next_state= chess_agent::Environment::SwitchBoardStatePov(cur_state);
                }

                game_replays.push_back(chess_agent::Replay(cur_state,
                                                        next_state,
                                                        action_prop_distr,
                                                        action_space,
                                                        ml_lib::Tensor::Scalar(0.)));
            }


            auto winner = env.get_active_player();
            auto cur_player = chess::Piece::Colour::White;

            // add replays to memory
            for(auto replay: game_replays) {
                replay.set_return(GenerateReturn(winner, cur_player));

                rm.Put(replay);

                cur_player = chess::Game::Opponent(cur_player);
            }

            // train actor and critic
            auto replay_batch = rm.GenerateRandomBatch<BATCHSIZE>();

            auto actor_out = ActorFeedForward(replay_batch.get_state(),
                                            replay_batch.get_action_space(),
                                            actor_model); // hier stoppt das programm!!

            auto critic_out = CriticFeedForward(replay_batch.get_state(),
                                            actor_out,
                                            critic_model);
                            
            {
                // prevent dead roots
                auto critic_loss = critic_lossfunc(critic_out,
                replay_batch.get_return());

                critic_optimizer.Step(critic_loss);
            }

            {
                // prevent dead roots
                auto actor_loss = critic_out.ScalarMult(ml_lib::Tensor::Scalar(-1.));

                actor_optimizer.Step(actor_loss);
            }
        }
    }
} // namespace chess_agent