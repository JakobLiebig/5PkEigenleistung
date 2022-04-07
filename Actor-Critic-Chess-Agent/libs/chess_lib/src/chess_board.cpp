#include "chess_lib/chess.h"

namespace chess
{
    const Piece::Colour Board::cPlayerAtTop = Piece::Colour::White;

    Board Board::BasicSetup()
    {
        std::array<Piece, 64> board_pieces;
        board_pieces.fill(Piece::Empty());

        // place white pieces
        board_pieces[0] = Piece::Rook(Board::cPlayerAtTop);
        board_pieces[1] = Piece::Knight(Board::cPlayerAtTop);
        board_pieces[2] = Piece::Bishop(Board::cPlayerAtTop);
        board_pieces[3] = Piece::Queen(Board::cPlayerAtTop);
        board_pieces[4] = Piece::King(Board::cPlayerAtTop);
        board_pieces[5] = Piece::Bishop(Board::cPlayerAtTop);
        board_pieces[6] = Piece::Knight(Board::cPlayerAtTop);
        board_pieces[7] = Piece::Rook(Board::cPlayerAtTop);

        board_pieces[8] = Piece::Pawn(Board::cPlayerAtTop);
        board_pieces[9] = Piece::Pawn(Board::cPlayerAtTop);
        board_pieces[10] = Piece::Pawn(Board::cPlayerAtTop);
        board_pieces[11] = Piece::Pawn(Board::cPlayerAtTop);
        board_pieces[12] = Piece::Pawn(Board::cPlayerAtTop);
        board_pieces[13] = Piece::Pawn(Board::cPlayerAtTop);
        board_pieces[14] = Piece::Pawn(Board::cPlayerAtTop);
        board_pieces[15] = Piece::Pawn(Board::cPlayerAtTop);

        // place black pieces
        Piece::Colour player_at_bottom = Game::Opponent(Board::cPlayerAtTop);
        board_pieces[63 - 0] = Piece::Rook(player_at_bottom);
        board_pieces[63 - 1] = Piece::Knight(player_at_bottom);
        board_pieces[63 - 2] = Piece::Bishop(player_at_bottom);
        board_pieces[63 - 3] = Piece::King(player_at_bottom);
        board_pieces[63 - 4] = Piece::Queen(player_at_bottom);
        board_pieces[63 - 5] = Piece::Bishop(player_at_bottom);
        board_pieces[63 - 6] = Piece::Knight(player_at_bottom);
        board_pieces[63 - 7] = Piece::Rook(player_at_bottom);

        board_pieces[63 - 8] = Piece::Pawn(player_at_bottom);
        board_pieces[63 - 9] = Piece::Pawn(player_at_bottom);
        board_pieces[63 - 10] = Piece::Pawn(player_at_bottom);
        board_pieces[63 - 11] = Piece::Pawn(player_at_bottom);
        board_pieces[63 - 12] = Piece::Pawn(player_at_bottom);
        board_pieces[63 - 13] = Piece::Pawn(player_at_bottom);
        board_pieces[63 - 14] = Piece::Pawn(player_at_bottom);
        board_pieces[63 - 15] = Piece::Pawn(player_at_bottom);

        return Board(board_pieces);
    }

    Board Board::MovePiece(const Move &m) const
    {
        int captured_points = 0;

        std::array<Piece, 64> next_state_board_pieces = m_pieces;

        Piece &moving_piece = next_state_board_pieces[m.m_from];
        Piece &captured_piece = next_state_board_pieces[m.m_to];

        if (m.isQueensideCastling(moving_piece))
        {
            int rook_position = m.m_from - 4;
            int rook_target_position = m.m_from - 1;
            Piece &rook = next_state_board_pieces[rook_position];

            next_state_board_pieces[rook_target_position] = rook;
            next_state_board_pieces[rook_position] = Piece::Empty();

            rook.IncrementStepsTaken();
        }
        else if (m.isKingsideCastling(moving_piece))
        {
            int rook_position = m.m_from + 3;
            int rook_target_position = m.m_from + 1;
            Piece &rook = next_state_board_pieces[rook_position];

            next_state_board_pieces[rook_target_position] = rook;
            next_state_board_pieces[rook_position] = Piece::Empty();

            rook.IncrementStepsTaken();
        }
        else if (m.isEnpassant(moving_piece, *this))
        {
            int captured_pawn_position = m.m_from + (m.m_to % 8) - (m.m_from % 8);
            Piece captured_pawn = next_state_board_pieces[captured_pawn_position];

            next_state_board_pieces[captured_pawn_position] = Piece::Empty();

            captured_points += captured_pawn.get_worth();
        }
        else if (m.isPawnPromotion(moving_piece))
        {
            moving_piece = Piece::Queen(moving_piece.get_colour());
        }

        // count captured_points
        captured_points += captured_piece.get_worth();

        // move piece
        moving_piece.IncrementStepsTaken();
        next_state_board_pieces[m.m_to] = moving_piece;
        next_state_board_pieces[m.m_from] = Piece::Empty();

        return Board(next_state_board_pieces);
    }
    std::vector<Move> Board::GeneratePseudoLegalMoves(const Piece::Colour &player) const
    {
        std::vector<Move> pseudo_legal_moves;

        unsigned int num_pieces = m_pieces.size();
        for (unsigned int piece_position = 0; piece_position < num_pieces; piece_position++)
        {
            const Piece &ith_piece = m_pieces[piece_position];

            if (ith_piece.get_colour() == player)
                ith_piece.GeneratePseudoLegalMoves(piece_position, pseudo_legal_moves, *this);
        }

        return pseudo_legal_moves;
    }

    bool Board::isKingUnderAttack(const Piece::Colour &king_owner, const std::vector<Move> &pseudo_legal_moves) const
    {
        const unsigned int num_pseudo_legal_moves = pseudo_legal_moves.size();
        for (unsigned int i = 0; i < num_pseudo_legal_moves; i++)
        {
            Move ith_pseudo_legal_move = pseudo_legal_moves[i];
            const Piece &piece_under_attack = m_pieces[ith_pseudo_legal_move.m_to];

            if (piece_under_attack.get_type() == Piece::Type::King && piece_under_attack.get_colour() == king_owner)
                return true;
        }

        return false;
    }

    std::string EncodePiece(const Piece &piece)
    {
        Piece::Colour piece_colour = piece.get_colour();

        switch (piece.get_type())
        {
        case Piece::Type::King:
            if (piece_colour == Piece::Colour::White)
                return "♚";
            else
                return "♔";
            break;

        case Piece::Type::Queen:
            if (piece_colour == Piece::Colour::White)
                return "♛";
            else
                return "♕";
            break;
        case Piece::Type::Rook:
            if (piece_colour == Piece::Colour::White)
                return "♜";
            else
                return "♖";
            break;
        case Piece::Type::Bishop:
            if (piece_colour == Piece::Colour::White)
                return "♝";
            else
                return "♗";
            break;
        case Piece::Type::Knight:
            if (piece_colour == Piece::Colour::White)
                return "♞";
            else
                return "♘";
            break;
        case Piece::Type::Pawn:
            if (piece_colour == Piece::Colour::White)
                return "♟︎";
            else
                return "♙";
            break;
        default:
            return " ";
            break;
        }
    }

    std::string Board::ToString() const
    {
        std::string piece_buffer = "  ";
        std::string cell_indexes = "   A" + piece_buffer + "B" + piece_buffer + "C" + piece_buffer + "D" + piece_buffer + "E" + piece_buffer + "F" + piece_buffer + "G" + piece_buffer + "H\n";
        
        std::string board_string = cell_indexes;

        for (unsigned int y = 0; y < 8; y++)
        {
            std::string row_string;
            for (unsigned int x = 0; x < 8; x++)
            {
                int field_index = x + y * 8;
                Piece piece_on_field = m_pieces[field_index];

                if(piece_on_field.get_type() == Piece::Type::Empty){
                // create checkered pattern
                    if (abs(field_index % 2 - y % 2))
                        row_string.append("." + piece_buffer);
                    else
                        row_string.append("_" + piece_buffer);
                } else {
                    row_string.append(EncodePiece(piece_on_field) + piece_buffer);
                }
            }

            std::string row_index = std::to_string(y + 1);
            board_string.append(row_index + "  " + row_string + " " + row_index + "\n");
        }

        board_string.append(cell_indexes);

        return board_string;
    }
    Piece Board::get_piece_at(const int &field_index) const
    {
        if (field_index < 0 || field_index >= 64)
            throw std::invalid_argument("field_index out of bounds");

        return m_pieces[field_index];
    }

    Board::Board(const std::array<Piece, 64> &pieces) : m_pieces(pieces)
    {
    }
} // namespace chess