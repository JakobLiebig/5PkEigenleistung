#include "chess_lib/chess.h"

namespace chess
{
    Game::Game() : m_active_player(Piece::Colour::White),
                   m_board(Board::BasicSetup()),
                   m_current_state_legal_moves()
    {
        m_current_state_legal_moves = GenerateLegalMoves();
    }

    bool Game::MovePiece(const Move &m)
    {
        // check if move is legal
        bool move_is_legal;
        unsigned int num_legal_moves = m_current_state_legal_moves.size();
        for (unsigned int i = 0; i < num_legal_moves; i++)
        {
            if (m_current_state_legal_moves[i] == m)
                move_is_legal = true;
        }

        if (!move_is_legal)
        {
            throw std::invalid_argument("Parameter needs to be a legal move!");
        }

        // move pieces
        m_board = m_board.MovePiece(m);

        // opponent becomes active player
        m_active_player = Opponent(m_active_player);

        // generate current state legal moves
        m_current_state_legal_moves = GenerateLegalMoves();

        // check if game is over
        if (m_current_state_legal_moves.size() == 0)
            return true;

        return false;
    }
    void Game::Reset()
    {
        m_board = Board::BasicSetup();
        m_active_player = Piece::Colour::White;

        m_current_state_legal_moves = GenerateLegalMoves();
    }

    Board Game::get_board() const
    {
        return m_board;
    }
    Piece::Colour Game::get_active_player() const
    {
        return m_active_player;
    }
    std::vector<Move> Game::get_legal_moves() const
    {
        return m_current_state_legal_moves;
    }

    Piece::Colour Game::Opponent(const Piece::Colour &active_player)
    {
        if (active_player == Piece::Colour::White)
            return Piece::Colour::Black;
        else
            return Piece::Colour::White;
    }


    std::vector<Move> Game::GenerateLegalMoves() const
    {
        std::vector<Move> pseudo_legal_moves = m_board.GeneratePseudoLegalMoves(m_active_player);
        std::vector<Move> legal_moves;

        unsigned int num_pseudo_legal_moves = pseudo_legal_moves.size();
        for (unsigned int i = 0; i < num_pseudo_legal_moves; i++)
        {
            Move ith_move = pseudo_legal_moves[i];
            Board ith_move_next_state = m_board.MovePiece(ith_move);

            Piece::Colour active_player_opponent = Game::Opponent(m_active_player);
            std::vector<Move> next_state_opponent_pseudo_legal_moves = ith_move_next_state.GeneratePseudoLegalMoves(active_player_opponent);

            if (!ith_move_next_state.isKingUnderAttack(m_active_player, next_state_opponent_pseudo_legal_moves))
                legal_moves.push_back(ith_move);
        }

        return legal_moves;
    }
} // namespace chess