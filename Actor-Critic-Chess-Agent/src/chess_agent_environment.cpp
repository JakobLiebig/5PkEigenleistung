#include "actor-critic-chess-agent/environment.h"

namespace chess_agent
{
    const chess::Piece::Colour Environment::cDefaultViewPoint = chess::Piece::Colour::White;
    const chess::Piece::Colour Environment::cFirstIdPieceColour = chess::Piece::Colour::White;

    Environment::Environment() : m_piece_positions(BasicPiecePositions()),
                                 m_game()
    {

    }

    int MirrorBoardPosition(const int &p)
    {
        int x_n = 8;
        int y_max_index = 7;

        int p_hat = 2 * (p % x_n) + x_n * y_max_index - p;
        return p_hat;
    }

    ml_lib::Tensor Environment::SwitchBoardStatePov(const ml_lib::Tensor &board_state) {
        // The agent is suppose to play against itself. That means he takes actions from pov.black and pov.white.
        // Normalize the input into the agent (meaning he always views the board from his pov)

        auto mirrored_board_state = board_state;

        for(unsigned int i = 0; i < board_state.get_num_elements(); i++) {
            auto coordinate_vector = board_state.IndexToPosition(i);
            
            int board_pos_x = coordinate_vector[0];
            int board_pos_y = coordinate_vector[1];

            int board_pos = board_pos_x + board_pos_y * 8;
            int mirrored_board_pos = MirrorBoardPosition(board_pos);

            int mirrored_board_pos_x = mirrored_board_pos % 8;
            int mirrored_board_pos_y = mirrored_board_pos / 8;

            coordinate_vector[0] = mirrored_board_pos_x;
            coordinate_vector[1] = mirrored_board_pos_y;

            int mirrored_index = board_state.PositionToIndex(coordinate_vector);
            
            mirrored_board_state.SetSingleElementValue(board_state.get_element_value_at(i), mirrored_index);
        }

        return mirrored_board_state;
    }

    chess::Move Environment::ActionPropDistrToMove(const ml_lib::Tensor &action_prop_distr)
    {
        double biggest_element;
        auto find_biggest_lambda = [&](const double &cur_value)
        {
            if (cur_value > biggest_element)
            {
                biggest_element = cur_value;
                return true;
            }
            return false;
        };

        unsigned int biggest_element_index = action_prop_distr.ArgFind(find_biggest_lambda);
        std::vector<unsigned int> biggest_element_coordinates = action_prop_distr.IndexToPosition(biggest_element_index);

        int to_x = biggest_element_coordinates[0];
        int to_y = biggest_element_coordinates[1];
        int to = to_x + to_y * 8;

        // a piece_id consists of the relative_id
        // the relative_id is the same for the same pieces of different colours
        // (piece_id of the black queen is the sam as the white queen)
        // the first 16 ids have the colour cFirstIdPieceColour (usually white)
        // thus if the piece_colour is black we have to add 16 to relative id
        
        int relative_piece_id = biggest_element_coordinates[2];
        int piece_id_extension = m_game.get_active_player() == cFirstIdPieceColour ? 0 : 16;
        int piece_id = piece_id_extension + relative_piece_id;
        
        int from = m_piece_positions[piece_id];

        // all action_spaces are normalized to pov of active_player
        // thus if ative_player is not default pov all positions need to be mirrored
        bool mirror = m_game.get_active_player() != cDefaultViewPoint;
        if(mirror)
            to = MirrorBoardPosition(to);

        return chess::Move(from, to);
    }
    ml_lib::Tensor Environment::MoveToActionPropDistr(const chess::Move &move)
    {
        // find piece_id
        // we only search through the pieces owned by active_player
        // if the colour is equal to cFirstIdPieceColour (usually white) the first id = 0
        // else 16
        int piece_id = 0;
        for(;;piece_id++) {
            if(m_piece_positions[piece_id] == move.m_from)
                break;
        }
        piece_id = (piece_id + 16) % 16;
        
        int to = move.m_to;
        
        // all action_spaces are normalized to pov of active_player
        // thus if ative_player is not default pov all positions need to be mirrored
        bool mirror = m_game.get_active_player() != cDefaultViewPoint;
        if(mirror)
            to = MirrorBoardPosition(to);

        auto action_prop_distr = ml_lib::Tensor::Zeros({8, 8, 16, 1});
        action_prop_distr.SetSingleElementValue(1., to + piece_id * 64);

        return action_prop_distr;
    }

    chess::Board Environment::get_board() const
    {
        return m_game.get_board();
    }
    std::vector<chess::Move> Environment::get_legal_moves() const {
        return m_game.get_legal_moves();
    }
    chess::Piece::Colour Environment::get_active_player() const
    {
        return m_game.get_active_player();
    }

    bool Environment::MovePiece(const chess::Move &move)
    {
        chess::Piece::Colour active_player = m_game.get_active_player();
        int first_player_piece_id = active_player == cFirstIdPieceColour ? 0 : 16;
        int first_opponent_piece_id = (first_player_piece_id + 16) % 32;

        // update piece_position for moving piece
        for (unsigned int i = first_player_piece_id; i - first_player_piece_id < 16; i++)
        {
            if (m_piece_positions[i] == move.m_from)
            {
                m_piece_positions[i] = move.m_to;
                break;
            }
        }

        // check for capture, move captured piece to -1
        for (unsigned int i = first_opponent_piece_id; i - first_opponent_piece_id < 16; i++)
        {
            if (m_piece_positions[i] == move.m_to)
            {
                m_piece_positions[i] = -1;
                break;
            }
        }

        return m_game.MovePiece(move);
    }
    void Environment::Reset()
    {
        m_piece_positions = BasicPiecePositions();

        m_game.Reset();
    }

    ml_lib::Tensor Environment::GenerateBoardState() const
    {
        ml_lib::Tensor state = ml_lib::Tensor::Zeros({8, 8, 16, 2, 1});

        for (unsigned int i = 0; i < m_piece_positions.size(); i++)
        {
            int piece_position = m_piece_positions[i];
            int piece_index = i;

            int piece_state_index = piece_position + piece_index * 64;

            state.SetSingleElementValue(piece_state_index, 1.);
        }

        return state;
    }
    ml_lib::Tensor Environment::GenerateActionSpace() const
    {
        // all action_spaces are normalized to pov of active_player
        // thus if ative_player is not default pov all positions need to be mirrored
        auto active_player = m_game.get_active_player();
        bool mirror = active_player != cDefaultViewPoint;

        // index of first piece which belongs to active_player
        int first_player_piece_id = active_player == cFirstIdPieceColour ? 0 : 16;

        auto action_space = ml_lib::Tensor::Zeros({8, 8, 16, 1});
        auto legal_moves = m_game.get_legal_moves();

        int cur_relatice_id, cur_from;
        cur_relatice_id = -1;
        cur_from = -1;

        for (chess::Move cur_legal_move : legal_moves)
        {
            int cur_to_board_pos = cur_legal_move.m_to;

            if (cur_legal_move.m_from != cur_from)
            {
                // if the legal_move from is not the same as before, the piece changed
                // thus the piece_id has changed and needs to be updated

                cur_from = cur_legal_move.m_from;

                for (int i = first_player_piece_id; i - first_player_piece_id < 16; i++)
                    // for all pieces which belong to the active player (first_piece_index till firstpiece_index + 16)
                    if (m_piece_positions[i] == cur_legal_move.m_from)
                    {
                        // find the index of the piece on cur_legal_move.m_from
                        cur_relatice_id = i;
                        break;
                    }
            }

            int cur_id = cur_relatice_id % 16;             

            if(mirror)
                cur_to_board_pos = MirrorBoardPosition(cur_to_board_pos);

            int cur_to_action_pos = cur_to_board_pos + cur_id * 64;
            action_space.SetSingleElementValue(1., cur_to_action_pos);
        }

        return action_space;
    }

    std::array<int, 32> Environment::BasicPiecePositions()
    {
        std::array<int, 32> basic_piece_positions;

        basic_piece_positions[0] = 0; // white rook
        basic_piece_positions[1] = 1; // white knight
        basic_piece_positions[2] = 2; // white bishop
        basic_piece_positions[3] = 3; // white queen
        basic_piece_positions[4] = 4; // white king
        basic_piece_positions[5] = 5; // white bishop
        basic_piece_positions[6] = 6; // white knight
        basic_piece_positions[7] = 7; // white rook

        basic_piece_positions[8] = 8;   // white pawn
        basic_piece_positions[9] = 9;   // white pawn
        basic_piece_positions[10] = 10; // white pawn
        basic_piece_positions[11] = 11; // white pawn
        basic_piece_positions[12] = 12; // white pawn
        basic_piece_positions[13] = 13; // white pawn
        basic_piece_positions[14] = 14; // white pawn
        basic_piece_positions[15] = 15; // white pawn

        basic_piece_positions[16] = 63 - 0; // black rook
        basic_piece_positions[17] = 63 - 1; // black knight
        basic_piece_positions[18] = 63 - 2; // black bishop
        basic_piece_positions[19] = 63 - 3; // black king
        basic_piece_positions[20] = 63 - 4; // black queen
        basic_piece_positions[21] = 63 - 5; // black bishop
        basic_piece_positions[22] = 63 - 6; // black knight
        basic_piece_positions[23] = 63 - 7; // black rook

        basic_piece_positions[24] = 63 - 8;  // black pawn
        basic_piece_positions[25] = 63 - 9;  // black pawn
        basic_piece_positions[26] = 63 - 10; // black pawn
        basic_piece_positions[27] = 63 - 11; // black pawn
        basic_piece_positions[28] = 63 - 12; // black pawn
        basic_piece_positions[29] = 63 - 13; // black pawn
        basic_piece_positions[30] = 63 - 14; // black pawn
        basic_piece_positions[31] = 63 - 15; // black pawn

        return basic_piece_positions;
    }
} // namespace chess_agent