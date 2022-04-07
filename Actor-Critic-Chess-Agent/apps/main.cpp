#include <iostream>
#include <vector>
#include <string>

#include "ml_lib/tensor.h"
#include "ml_lib/model.h"
#include "ml_lib/replay_memory.h"

#include "actor-critic-chess-agent/environment.h"

#define LOG(x) std::cout << x << std::endl
    ml_lib::Tensor chess_agent::ActorFeedForward(const ml_lib::Tensor& board_state, const ml_lib::Tensor& action_space, std::vector<ml_lib::LayerBase*> actor_model) 
    {
        // board_state shape = {8, 8, 16, 2}
        auto out = board_state;
        
        unsigned int batchsize = out.get_num_elements() / 2048;
        out = out.Reshape({2048, batchsize});

        for(unsigned int i = 0; i < actor_model.size() -1; i++) {
            out = actor_model[i]->FeedForward(out);
        }

        out = out.HadamardMult(action_space);

        //out = actor_model.back()->FeedForward(out);

        out = out.Reshape({8, 8, 16, batchsize});

        return out;
    }
   
class Test {
public:
    Test():
        m_t(ml_lib::Tensor::Zeros({2, 1})) {
    }

    Test(const ml_lib::Tensor& t):
        m_t(t) {
    }

    void Out() {
        m_t.LogElementValues();
    }

    static Test Concatenate(const Test& a, const Test& b) {
        auto t = ml_lib::Tensor::Concatenate(a.m_t, b.m_t, 1);

        return Test(t);
    }

private:
    ml_lib::Tensor m_t;
};

int main() {
    // actor
    ml_lib::layer_type::Linear actor_l1(2048, 1024, ml_lib::initializer::Jakob, ml_lib::initializer::Jakob);
    ml_lib::layer_type::Linear actor_l2(1024, 1024, ml_lib::initializer::Jakob, ml_lib::initializer::Jakob);
    ml_lib::layer_type::Linear actor_l3(1024, 1024, ml_lib::initializer::Jakob, ml_lib::initializer::Jakob);
    ml_lib::layer_type::Softmax actor_norm(2);
    std::vector<ml_lib::LayerBase*> actor_model = {&actor_l1, &actor_l2, &actor_l3};

    chess_agent::train(2, actor_model);
    chess_agent::test(actor_model);

    return 0;
}