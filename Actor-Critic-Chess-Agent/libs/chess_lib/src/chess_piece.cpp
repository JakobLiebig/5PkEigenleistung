#include "chess_lib/chess.h"

namespace chess
{
    Piece Piece::Empty()
    {
        return Piece();
    }
    Piece Piece::King(const Colour &colour)
    {
        return Piece(Piece::Type::King, 0U, colour);
    }
    Piece Piece::Queen(const Colour &colour)
    {
        return Piece(Piece::Type::Queen, 9U, colour);
    }
    Piece Piece::Bishop(const Colour &colour)
    {
        return Piece(Piece::Type::Bishop, 0U, colour);
    }
    Piece Piece::Knight(const Colour &colour)
    {
        return Piece(Piece::Type::Knight, 3U, colour);
    }
    Piece Piece::Rook(const Colour &colour)
    {
        return Piece(Piece::Type::Rook, 4U, colour);
    }
    Piece Piece::Pawn(const Colour &colour)
    {
        return Piece(Piece::Type::Pawn, 1U, colour);
    }

    Piece::Piece() : m_colour(Piece::Colour::None),
                     m_steps_taken(0U),
                     m_type(Piece::Type::Empty),
                     m_worth(0U)
    {
    }

    void Piece::GeneratePseudoLegalMoves(const unsigned int &starting_field,
                                         std::vector<Move> &pseudo_legal_moves,
                                         const Board &board) const
    {
        switch (m_type)
        {
        case Piece::Type::King:
            GenerateKingMoves(starting_field, pseudo_legal_moves, board);
            break;

        case Piece::Type::Queen:
            GenerateQueenMoves(starting_field, pseudo_legal_moves, board);
            break;

        case Piece::Type::Bishop:
            GenerateBishopMoves(starting_field, pseudo_legal_moves, board);
            break;

        case Piece::Type::Knight:
            GenerateKnightMoves(starting_field, pseudo_legal_moves, board);
            break;

        case Piece::Type::Rook:
            GenerateRookMoves(starting_field, pseudo_legal_moves, board);
            break;

        case Piece::Type::Pawn:
            GeneratePawnMoves(starting_field, pseudo_legal_moves, board);
            break;

        default:
            break;
        }
    }

    void Piece::IncrementStepsTaken()
    {
        m_steps_taken += 1U;
    }

    Piece::Colour Piece::get_colour() const
    {
        return m_colour;
    }
    Piece::Type Piece::get_type() const
    {
        return m_type;
    }
    unsigned int Piece::get_worth() const
    {
        return m_worth;
    }

    Piece::Piece(const Piece::Type &type, const unsigned int &worth, const Colour &colour) : m_colour(colour),
                                                                                             m_steps_taken(0U),
                                                                                             m_type(type),
                                                                                             m_worth(0U)
    {
    }

    void Piece::ContinueGeneratingSlidingMoves(const Move &first_move,
                                               std::vector<Move> &pseudo_legal_moves,
                                               const Board &board) const
    {
        const int cFirstMoveDirection = first_move.m_to - first_move.m_from;
        const int cFirstMoveDistance = abs(cFirstMoveDirection);

        int current_field = first_move.m_from;

        while (true)
        {
            current_field += cFirstMoveDirection;
            Move current_move(first_move.m_from, current_field);

            bool is_straight_slide = (cFirstMoveDistance == 1 || cFirstMoveDistance == 8) &&
                                     current_move.isStraightSlide();

            bool is_diagonal_slide = (cFirstMoveDistance != 1 && cFirstMoveDistance != 8) &&
                                     current_move.isDiagonalSlide();

            if (!is_straight_slide && !is_diagonal_slide)
                return;

            Piece piece_on_target_field = board.get_piece_at(current_move.m_to);

            if (piece_on_target_field.get_type() == Piece::Type::Empty)
                // no piece on current position
                pseudo_legal_moves.push_back(current_move);

            else if (piece_on_target_field.get_colour() == m_colour)
                // piece belongs to current player
                return;

            else if (piece_on_target_field.get_colour() != Piece::Colour::None)
            {
                // piece belongs to other player and can be captured
                pseudo_legal_moves.push_back(current_move);
                return;
            }
        }
    }

    void Piece::GenerateKingMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const
    {
        // default king movement
        static const std::array<int, 8> cDefaultMoveDirections = {1, -1, 9, -9, 8, -8, 7, -7};
        static const unsigned int num_default_move_directions = cDefaultMoveDirections.size();

        for (unsigned int i = 0; i < num_default_move_directions; i++)
        {
            Move ith_default_move(starting_field, starting_field + cDefaultMoveDirections[i]);

            if (!ith_default_move.isDiagonalSlide() &&
                !ith_default_move.isStraightSlide())
                continue;

            Piece piece_on_target_field = board.get_piece_at(ith_default_move.m_to);

            if (piece_on_target_field.get_colour() != m_colour)
                pseudo_legal_moves.push_back(ith_default_move);
        }

        if (m_steps_taken == 0U)
        {
            // kingside castling
            Move kingside_castling(starting_field, starting_field + 2);
            Move move_to_kingside_rook(starting_field, starting_field + 3);

            if (isCastlingPossible(move_to_kingside_rook, board))
                pseudo_legal_moves.push_back(kingside_castling);

            // queenside castling
            Move queenside_castling(starting_field, starting_field - 2);
            Move move_to_queenside_rook(starting_field, starting_field - 4);

            if (isCastlingPossible(move_to_queenside_rook, board))
                pseudo_legal_moves.push_back(queenside_castling);
        }
    }
    void Piece::GenerateQueenMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const
    {
        static const std::array<int, 8> cMoveDirections = {1, -1, 9, -9, 8, -8, 7, -7};
        static const unsigned int num_move_directions = cMoveDirections.size();

        for (unsigned int i = 0; i < num_move_directions; i++)
        {
            Move ith_direction_first_move(starting_field, starting_field + cMoveDirections[i]);

            ContinueGeneratingSlidingMoves(ith_direction_first_move, pseudo_legal_moves, board);
        }
    }
    void Piece::GenerateBishopMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const
    {
        static const std::array<int, 4> cMoveDirections = {9, -9, 7, -7};
        static const unsigned int num_move_directions = cMoveDirections.size();

        for (unsigned int i = 0; i < num_move_directions; i++)
        {
            Move ith_direction_first_move(starting_field, starting_field + cMoveDirections[i]);

            ContinueGeneratingSlidingMoves(ith_direction_first_move, pseudo_legal_moves, board);
        }
    }
    void Piece::GenerateKnightMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const
    {
        // 6 doesnt work fpr m_start: 0
        static const std::array<int, 8> cMoveDirections = {6, -6, 10, -10, 15, -15, 17, -17};
        static const unsigned int num_move_directions = cMoveDirections.size();

        for (unsigned int i = 0; i < num_move_directions; i++)
        {
            Move ith_move(starting_field, starting_field + cMoveDirections[i]);

            if (!isKnightMove(ith_move))
                continue;

            Piece piece_at_target_field = board.get_piece_at(ith_move.m_to);

            if (piece_at_target_field.get_colour() != m_colour)
                pseudo_legal_moves.push_back(ith_move);
        }
    }
    void Piece::GenerateRookMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const
    {
        static const std::array<int, 4> cMoveDirections = {-1, 1, 8, -8};
        static const unsigned int num_move_directions = cMoveDirections.size();

        for (unsigned int i = 0; i < num_move_directions; i++)
        {
            Move ith_direction_first_move(starting_field, starting_field + cMoveDirections[i]);

            ContinueGeneratingSlidingMoves(ith_direction_first_move, pseudo_legal_moves, board);
        }
    }
    void Piece::GeneratePawnMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const
    {
        int advancing_direction = m_colour == Board::cPlayerAtTop ? 8 : -8;
        Piece::Colour active_players_opponent = Game::Opponent(m_colour);

        int opponent_double_pawn_opening_row = m_colour == Board::cPlayerAtTop ? 4 : 3;
        int starting_field_row = starting_field / 8;

        // possible moves:
        Move advance_one(starting_field, starting_field + advancing_direction);
        Move double_opening(starting_field, starting_field + 2 * advancing_direction);
        Move capture_left(starting_field, starting_field + advancing_direction - 1);
        Move capture_right(starting_field, starting_field + advancing_direction + 1);
        Move &left_en_passant = capture_left;
        Move &right_en_passant = capture_right;

        // advance_one
        // cannot leave board boundaries due to pawn promotion
        Piece piece_one_ahead = board.get_piece_at(advance_one.m_to);

        if (piece_one_ahead.get_type() == Piece::Type::Empty)
        {
            pseudo_legal_moves.push_back(advance_one);

            // double opening

            if (m_steps_taken == 0U)
            {
                Piece piece_two_ahead = board.get_piece_at(double_opening.m_to);

                if (piece_two_ahead.get_type() == Piece::Type::Empty)
                {
                    pseudo_legal_moves.push_back(double_opening);
                }
            }
        }

        if (capture_right.isDiagonalSlide())
        {
            Piece captured_piece_right = board.get_piece_at(capture_right.m_to);

            // capture right
            if (capture_right.isCapture(board))
                pseudo_legal_moves.push_back(capture_right);
            else if (captured_piece_right.get_type() == Piece::Type::Empty)
            {
                // en passant right
                Piece en_passant_right_captured_piece = board.get_piece_at(starting_field + 1);

                if (en_passant_right_captured_piece.get_colour() == active_players_opponent &&
                    en_passant_right_captured_piece.m_steps_taken == 1U &&
                    starting_field_row == opponent_double_pawn_opening_row)
                    pseudo_legal_moves.push_back(right_en_passant);
            }
        }
    
        if (capture_left.isDiagonalSlide())
        {
            Piece captured_piece_left = board.get_piece_at(capture_left.m_to);

            // capture left
            if (capture_left.isCapture(board))
                pseudo_legal_moves.push_back(capture_left);
            else if (captured_piece_left.get_type() == Piece::Type::Empty)
            {
                // en passant left
                Piece en_passant_left_captured_piece = board.get_piece_at(starting_field - 1);

                if (en_passant_left_captured_piece.get_colour() == active_players_opponent &&
                    en_passant_left_captured_piece.m_steps_taken == 1U &&
                    starting_field_row == opponent_double_pawn_opening_row)
                    pseudo_legal_moves.push_back(left_en_passant);
            }
        }
    }

    bool Piece::isCastlingPossible(const Move &move_to_rook, const Board &board) const
    {
        Piece rook = board.get_piece_at(move_to_rook.m_to);
        bool castling_possible = rook.get_type() == Piece::Type::Rook && rook.m_steps_taken == 0U;

        int direction_to_rook = move_to_rook.m_to - move_to_rook.m_from;
        int single_step_towards_rook_direction = direction_to_rook >= 0 ? 1 : -1;

        unsigned int i = move_to_rook.m_from + single_step_towards_rook_direction;
        for (; i != move_to_rook.m_to && castling_possible; i += single_step_towards_rook_direction)
        {
            castling_possible = board.get_piece_at(i).get_type() == Piece::Type::Empty;
        }

        return castling_possible;
    }

    bool Piece::isKnightMove(const Move &move)
    {
        if (move.LeavesBoardBoundaries())
            return false;

        int x_distance = abs(move.m_to % 8 - move.m_from % 8);
        int y_distance = abs(move.m_to / 8 - move.m_from / 8);

        if ((x_distance == 2 && y_distance == 1) ||
            (x_distance == 1 && y_distance == 2))
            return true;

        return false;
    }
} // namespace chess