#ifndef CHESS_HEADER_GUARD
#define CHESS_HEADER_GUARD

#include <stdexcept>
#include <vector>
#include <array>
#include <string>
#include <iostream>

namespace chess
{
    class Board;
    class Piece;
    class Game;

    struct Move
    {
    public:
        Move(const std::string& from, const std::string& to);
        Move(const int &from = 0, const int &to = 0);
        bool operator==(const Move &other) const;

        bool LeavesBoardBoundaries() const;
        bool isStraightSlide() const;
        bool isDiagonalSlide() const;
        bool isCapture(const Board &board) const;

        bool isQueensideCastling(const Piece &moving_piece) const;
        bool isKingsideCastling(const Piece &moving_piece) const;
        bool isEnpassant(const Piece &moving_piece, const Board &board) const;
        bool isPawnPromotion(const Piece &moving_piece) const;

        int m_from;
        int m_to;
    };

    class Piece
    {
    public:
        enum class Colour
        {
            White,
            Black,
            None
        };

        enum class Type
        {
            Empty,
            King,
            Queen,
            Bishop,
            Knight,
            Rook,
            Pawn
        };

        static Piece Empty();
        static Piece King(const Colour &colour);
        static Piece Queen(const Colour &colour);
        static Piece Bishop(const Colour &colour);
        static Piece Knight(const Colour &colour);
        static Piece Rook(const Colour &colour);
        static Piece Pawn(const Colour &colour);

        Piece();

        void GeneratePseudoLegalMoves(const unsigned int &starting_field,
                                      std::vector<Move> &pseudo_legal_moves,
                                      const Board &board) const;

        void IncrementStepsTaken();

        Colour get_colour() const;
        Piece::Type get_type() const;
        void reset_type(const Piece::Type &new_type);
        unsigned int get_worth() const;

    private:
        Piece(const Piece::Type &type, const unsigned int &worth, const Colour &colour);

        void ContinueGeneratingSlidingMoves(const Move &first_move,
                                            std::vector<Move> &pseudo_legal_moves,
                                            const Board &board) const;

        void GenerateKingMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const;
        void GenerateQueenMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const;
        void GenerateBishopMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const;
        void GenerateKnightMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const;
        void GenerateRookMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const;
        void GeneratePawnMoves(const unsigned int &starting_field, std::vector<Move> &pseudo_legal_moves, const Board &board) const;

        bool isCastlingPossible(const Move &move_to_rook, const Board &board) const;
        static bool isKnightMove(const Move& move);

        Colour m_colour;
        unsigned int m_steps_taken;
        Piece::Type m_type;
        unsigned int m_worth;
    };

    class Board
    {
    public:
        const static Piece::Colour cPlayerAtTop;

        static Board BasicSetup();

        Board MovePiece(const Move &m) const;
        std::vector<Move> GeneratePseudoLegalMoves(const Piece::Colour &player) const;

        bool isKingUnderAttack(const Piece::Colour &king_owner, const std::vector<Move> &opponent_legal_moves) const;

        std::string ToString() const;
        Piece get_piece_at(const int &field_index) const;

    private:
        Board(const std::array<Piece, 64> &b);

        std::array<Piece, 64> m_pieces;
    };

    class Game
    {
    public:
        Game();

        bool MovePiece(const Move &m);
        void Reset();
        
        Board get_board() const;
        Piece::Colour get_active_player() const;
        std::vector<Move> get_legal_moves() const;

        static Piece::Colour Opponent(const Piece::Colour &active_player);
    private:
        std::vector<Move> GenerateLegalMoves() const;

        Piece::Colour m_active_player;
        Board m_board;

        std::vector<Move> m_current_state_legal_moves;
    };
} // namespace chess

#endif //!CHESS_HEADER_GUARD