
import numpy as np

from DancingLinksX import DancingLinksX


class SudokuSolver(object):

    def __init__(self,matrix):
        self.sudoku_matrix = matrix
        self.dlx = DancingLinksX(self.sudoku_matrix)
        self.solution_board = np.zeros(shape=(9,9),dtype=np.int32)

    def solve_sudoku(self):
        solutions_from_sudoku = self.dlx.search(0)
        return solutions_from_sudoku


    def final_solution(self,solutions):
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



            for r in rows_from_sparse_matrix:
                row_and_column = r[0:81]
                value = r[81:162]
                one = row_and_column.index(1)
                row = one / 9
                column = one % 9
                one = value.index(1)
                v = (one % 9) + 1
                self.solution_board[row][column] = v

