import copy
from Queue import Queue
#klasa koja reprezentuje problem koji se resava
from State import State


class Sudoku(object):
    def __init__(self,sudoku_board):
        self.sudoku_board = sudoku_board
        self.allowed_numbers = range(1, 10)
        self.total_sum = sum(range(1,10))


    #metoda koja vraca prvu slobodnu celiju na tabli,kako bi se sledeci potez odigrao u njoj
    def empty_cell(self,board):
        for row in xrange(len(board)):
            for column in xrange(len(board[row])):
                if board[row][column] == 0:
                    return (row,column)
        return (-1,-1)

    def determine_allowed_moves(self,board):
        allowed_moves = []

        #pronadji prvu slobodnu celiju
        row,column = self.empty_cell(board)
        allowed = self.allowed_numbers

        #proveri da li taj broj moze da se nadje u redu
        numbers_from_row = self.numbers_from_row(row,board)
        #izbaci sve one brojeve koji su u self.allowed_numbers kako se ne bi pojavili u tabli opet
        allowed = self.good_numbers(allowed,numbers_from_row)

        numbers_from_column = self.numbers_from_column(column,board)
        allowed = self.good_numbers(allowed,numbers_from_column)

        numbers_from_box = self.numbers_from_box(row,column,board)
        allowed = self.good_numbers(allowed,numbers_from_box)

        for number in allowed:
            allowed_move = AllowedMove(row,column,number)
            allowed_moves.append(allowed_move)

        return allowed_moves



    #funkcija koja vraca sve brojeve koji su u u redu kojeg oznacava parametar -> row
    def numbers_from_row(self,row,board):
       return  [number for number in board[row] if number != 0]

    # funkcija koja vraca sve brojeve koji su u u koloni koju oznacava parametar -> column
    def numbers_from_column(self,column,board):
       numbers = []
       for row in xrange(len(board)):
           if board[row][column] != 0:
               numbers.append(board[row][column])
       return numbers


    # funkcija koja vraca sve brojeve koji su u malom regionu koji se dobija parametrima -> row i column
    def numbers_from_box(self,row,column,board):
        box_region_start_row = row - (row % 3)
        box_region_start_column = column - (column % 3)
        numbers = []
        for i in xrange(0, 3):
            for j in xrange(0, 3):
                if board[box_region_start_row + i][box_region_start_column + j] != 0:
                    numbers.append(board[box_region_start_row + i][box_region_start_column + j])
        return numbers


    def good_numbers(self,first_list,second_list):
        return [number for number in first_list if number not in second_list]


    def update_board(self,board,allowed_move):

        new_board = copy.deepcopy(board)
        new_board[allowed_move.row][allowed_move.column] = allowed_move.number
        return new_board

    #provera da li je igra gotova
    def game_finished(self,board):

        #proveri za svaki red da li je suma tacno onolika kolika je trebala da bude
        for row in xrange(len(board)):
            if sum(board[row]) != self.total_sum:
                return False

        for row in xrange(len(board)):
            total_for_column = 0
            for column in xrange(len(board[row])):
                total_for_column += board[row][column]
            if total_for_column != self.total_sum:
                return False


        for row in xrange(0,9,3):
            for column in xrange(0,9,3):

                total_for_block = 0
                #mini matrica
                for block_row in xrange(0,3):
                    for block_column in xrange(0,3):

                        total_for_block += board[block_row + row][block_column + column]
                if total_for_block != self.total_sum:
                    return False
        return True


    def print_sudoku_board(self,board):
        for row in xrange(9):
            for column in xrange(9):
                print "\t" + str(board[row][column]),
            print("\n")




class BFS(object):


    def bfsSudoku(self,sudoku):
        #pocetno stanje
        state = State(sudoku.sudoku_board,None)

        #ako je zavrseno vracamo to stanje i tablu koju on nosi sa sobom
        if sudoku.game_finished(state.board):
            return state

        sudoku_queue = Queue()
        sudoku_queue.put(state)

        while sudoku_queue.qsize() != 0:

            state = sudoku_queue.get()
            for state_child in state.generate_next_states(sudoku):
                if (sudoku.game_finished(state_child.board)):
                    # ako je zavrseno vracamo to stanje i tablu koju on nosi sa sobom
                    return state_child
                sudoku_queue.put(state_child)


        return None


class AllowedMove(object):

    def __init__(self,row,column,number):
        self.row = row
        self.column = column
        self.number = number


