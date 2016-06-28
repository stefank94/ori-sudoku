from Sudoku import Sudoku, BFS

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

    #Resavanje sudoku igre preko BFS algoritma
    sudoku = Sudoku(test_matrix)
    bfs_solver = BFS()
    winning_state = bfs_solver.bfsSudoku(sudoku)
    sudoku.print_sudoku_board(winning_state.board)