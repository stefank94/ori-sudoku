
import numpy as np

from DancingLinksX import DancingLinksX


class SudokuSolver(object):

    def __init__(self,matrix):
        self.sudoku_matrix = matrix
        self.dlx = DancingLinksX(self.sudoku_matrix)

    def solve_sudoku(self):
        solutions_from_sudoku = self.dlx.search(0)
        return solutions_from_sudoku


    def print_solution(self,solutions):
        if len(solutions) == 0 :
            print("Nije pronadjeno resenje")
        else:
            solution_rows = []
            for node in solutions:
              solution_rows.append(node.row)

            rows_from_sparse_matrix = []
            for r in solution_rows:
                row_from_sparse_matrix = self.dlx.sparse_matrix[r]
                rows_from_sparse_matrix.append(row_from_sparse_matrix)

            solution_board = np.zeros(shape=(9,9),dtype=np.int32)

            for r in rows_from_sparse_matrix:
                row_and_column = r[0:81]
                value = r[81:162]
                one = row_and_column.index(1)
                row = one / 9
                column = one % 9
                one = value.index(1)
                v = (one % 9) + 1
                solution_board[row][column] = v

            for r in xrange(len(solution_board)):
                for c in xrange(len(solution_board[r])):
                    print "\t" + str(solution_board[r][c]) ,

                print ("\n")


if __name__ == '__main__':
    test_matrix =  [[8, 9, 2, 0, 0, 3, 0, 1, 4],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 6, 8, 0, 7, 0],
                   [4, 5, 0, 0, 8, 0, 0, 0, 1],
                   [0, 0, 8, 0, 0, 0, 2, 0, 0],
                   [1, 0, 3, 7, 0, 0, 5, 0, 0],
                   [0, 7, 1, 0, 0, 6, 0, 5, 0],
                   [5, 0, 9, 2, 0, 0, 0, 8, 0],
                   [6, 0, 0, 0, 0, 7, 0, 0, 9]]
    sudoku_solver = SudokuSolver(test_matrix)
    solutions = sudoku_solver.solve_sudoku()
    sudoku_solver.print_solution(solutions)


