#include "chess_lib/chess.h"

namespace chess
{
    Move::Move(const std::string &from, const std::string &to)
    {
        int from_x = (int)from[0] - (int)'a';
        int from_y = (int)from[1] - (int)'1';
        m_from = from_x + from_y * 8;

        int to_x = (int)to[0] - (int)'a';
        int to_y = (int)to[1] - (int)'1';
        m_to = to_x + to_y * 8;
    }

    Move::Move(const int &from, const int &to) : m_from(from),
                                                 m_to(to)
    {
    }

    bool Move::operator==(const Move &other) const
    {
        return (m_from == other.m_from && m_to == other.m_to);
    }

    bool Move::LeavesBoardBoundaries() const
    {
        if (m_from >= 64 || m_from < 0 || m_to >= 64 || m_to < 0)
            return true;

        return false;
    }
    bool Move::isStraightSlide() const
    {
        if(LeavesBoardBoundaries())
            return false;

        int x_direction = m_to % 8 - m_from % 8;
        int y_direction = m_to / 8 - m_from / 8;

        if (x_direction == 0 || y_direction == 0)
            return true;

        return false;
    }
    bool Move::isDiagonalSlide() const
    {
        if(LeavesBoardBoundaries())
            return false;
        
        int x_direction = m_to % 8 - m_from % 8;
        int y_direction = m_to / 8 - m_from / 8;

        if (abs(x_direction) == abs(y_direction))
            return true;

        return false;
    }
    bool Move::isCapture(const Board &board) const
    {
        if(LeavesBoardBoundaries())
            return false;
        
        Piece moving_piece = board.get_piece_at(m_from);
        Piece captured_piece = board.get_piece_at(m_to);

        if (captured_piece.get_colour() == Game::Opponent(moving_piece.get_colour()))
            return true;

        return false;
    }

    bool Move::isQueensideCastling(const Piece &moving_piece) const
    {
        int direction = m_to - m_from;

        return moving_piece.get_type() == Piece::Type::King && direction == -2;
    }
    bool Move::isKingsideCastling(const Piece &moving_piece) const
    {
        int direction = m_to - m_from;

        return moving_piece.get_type() == Piece::Type::King && direction == 2;
    }
    bool Move::isEnpassant(const Piece &moving_piece, const Board &board) const
    {
        int abs_direction = abs(m_to - m_from);

        return moving_piece.get_type() == Piece::Type::Pawn && !isCapture(board) && (abs_direction == 9 || abs_direction == 7);
    }
    bool Move::isPawnPromotion(const Piece &moving_piece) const
    {
        unsigned int target_row = m_to / 8;

        return moving_piece.get_type() == Piece::Type::Pawn && (target_row == 0 || target_row == 7);
    }
} // namespace chess