#klasa koja reprezentuje jedno stanje (cvor) u stablu pretrage
class State(object):

    def __init__(self,board,allowed_move= None):
        self.board = board
        self.allowed_move = allowed_move


    def new_state(self,sudoku,allowed_move):
        new_board = sudoku.update_board(self.board,allowed_move)
        return State(new_board,allowed_move)

    def generate_next_states(self,sudoku):
        next_states = []
        for allowed_move in sudoku.determine_allowed_moves(self.board):
            next_state = self.new_state(sudoku,allowed_move)
            next_states.append(next_state)
        return next_states