import numpy as np
class BacktrackingSudokuSolver(object):
    # Algoritam Backtracking
    # najcesci algoritam za resavanje sudoku problema.Garantuje ispravno resenje
    # https://en.wikipedia.org/wiki/Backtracking
    def __init__(self,sudoku_matrix):
        self.sudoku_matrix  = sudoku_matrix

    def solve(self):
        cell = self.exists_unssigned_cells();
        if cell == (-1,-1):
            return True
        else:
            for i in xrange(1,10):
                if not self.check_if_taken(i,cell[0],cell[1]):
                    self.sudoku_matrix[cell[0]][cell[1]] = i
                    if self.solve():
                        return True
                    self.sudoku_matrix[cell[0]][cell[1]] = 0
            return False

    def exists_unssigned_cells(self):
        for row in xrange(len(self.sudoku_matrix)):
            for column in xrange (len(self.sudoku_matrix[row])):
                if self.sudoku_matrix[row][column] == 0:
                    return (row,column)
        return (-1,-1)

    def check_if_taken(self,digit,row,column):
        #provera da li postoji u redovima
        if (digit in self.sudoku_matrix[row]):
            return True
        #provera da li posotji u kolonama
        for r in xrange(len(self.sudoku_matrix)):
            if self.sudoku_matrix[r][column] == digit:
                return True
        #provera da li postoji u jednom regionu
        box_region_start_row = row - (row %3)
        box_region_start_column = column - (column %3)
        for i in xrange(0,3):
            for j in xrange(0,3):
                if self.sudoku_matrix[box_region_start_row+i][box_region_start_column+j] == digit:
                    return True
        return False

    def print_matrix(self):
        for row in xrange(9):
            for column in xrange(9):
                print "\t" + str(self.sudoku_matrix[row][column]),
            print("\n")


if __name__ == '__main__':
    test_matrix = [[8, 9, 2, 0, 0, 3, 0, 1, 4],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 6, 8, 0, 7, 0],
                   [4, 5, 0, 0, 8, 0, 0, 0, 1],
                   [0, 0, 8, 0, 0, 0, 2, 0, 0],
                   [1, 0, 3, 7, 0, 0, 5, 0, 0],
                   [0, 7, 1, 0, 0, 6, 0, 5, 0],
                   [5, 0, 9, 2, 0, 0, 0, 8, 0],
                   [6, 0, 0, 0, 0, 7, 0, 0, 9]]
    backtracking_solver = BacktrackingSudokuSolver(test_matrix)
    backtracking_solver.solve()
    backtracking_solver.print_matrix()