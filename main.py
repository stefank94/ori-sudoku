import network
from Tkinter import *
import tkFileDialog
from SudokuSolverBacktracking import BacktrackingSudokuSolver
import copy



MARGIN = 20
SIDE = 50  # Sirrina celije
WIDTH = HEIGHT = MARGIN * 2 + SIDE * 9  # Sirina i visina cele table


class SudokuUI(Frame):
    def __init__(self, parent, game):
        self.game = game
        Frame.__init__(self, parent)
        self.parent = parent

        self.row, self.col = -1, -1

        self.__initUI()

    def __initUI(self):
        self.parent.title("Sudoku")
        self.pack(fill=BOTH, expand = YES)
        self.canvas = Canvas(self,
                             width=WIDTH,
                             height=HEIGHT)
        self.canvas.pack(fill=BOTH, side=TOP)
        
        menubar = Menu(self.parent)
        
        # create a pulldown menu, and add it to the menu bar
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.__open)
        menubar.add_cascade(label="File", menu=filemenu)        

        self.parent.config(menu=menubar)    
        
        solve_button = Button(self,text= "Solve sudoku",
                              command = self.__solve_sudoku)
                              
        solve_button.pack(fill= X,side = LEFT, anchor = W, expand = YES)
        
        
        clear_button = Button(self,
                              text="Clear answers",
                              command=self.__clear_answers)
        clear_button.pack(fill= X, side=LEFT, anchor = W, expand = YES)
        
       

        self.__draw_grid()
        self.__draw_puzzle()

        self.canvas.bind("<Button-1>", self.__cell_clicked)
        self.canvas.bind("<Key>", self.__key_pressed)
        

    def __open(self):
        self.file_opt = options = {}
        options['defaultextension'] = '.jpg'
        options['filetypes'] = [('image files', '.jpg'),('all files', '.*') ]
        options['initialdir'] = 'C:\\'
        self.file_path = tkFileDialog.askopenfilename(**self.file_opt)
        matrix = network.predict(self.file_path)
        self.game.start_puzzle = matrix
        self.game.start()
        self.__clear_answers()


    def __draw_grid(self):
        """
        Draws grid divided with blue lines into 3x3 squares
        """
        for i in xrange(10):
            color = "blue" if i % 3 == 0 else "gray"

            x0 = MARGIN + i * SIDE
            y0 = MARGIN
            x1 = MARGIN + i * SIDE
            y1 = HEIGHT - MARGIN
            self.canvas.create_line(x0, y0, x1, y1, fill=color)

            x0 = MARGIN
            y0 = MARGIN + i * SIDE
            x1 = WIDTH - MARGIN
            y1 = MARGIN + i * SIDE
            self.canvas.create_line(x0, y0, x1, y1, fill=color)

    def __draw_puzzle(self,solved_puzzle=None):
        self.canvas.delete("numbers")
        for i in xrange(9):
            for j in xrange(9):
                if solved_puzzle is not None:
                    answer = solved_puzzle[i][j]
                    color = self.determine_solved_numbers(i,j)
                    x = MARGIN + j * SIDE + SIDE / 2
                    y = MARGIN + i * SIDE + SIDE / 2
                    self.canvas.create_text(x, y, text=answer, tags="numbers", fill=color)
                else:
                    answer = self.game.puzzle[i][j]
                    if answer != 0:
                        x = MARGIN + j * SIDE + SIDE / 2
                        y = MARGIN + i * SIDE + SIDE / 2
                        color = "black"
                        self.canvas.create_text(
                            x, y, text=answer, tags="numbers", fill=color)
                        

    def __draw_cursor(self):
        self.canvas.delete("cursor")
        if self.row >= 0 and self.col >= 0:
            x0 = MARGIN + self.col * SIDE + 1
            y0 = MARGIN + self.row * SIDE + 1
            x1 = MARGIN + (self.col + 1) * SIDE - 1
            y1 = MARGIN + (self.row + 1) * SIDE - 1
            self.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline="red", tags="cursor"
            )

    def __cell_clicked(self, event):
        x, y = event.x, event.y
        if (MARGIN < x < WIDTH - MARGIN and MARGIN < y < HEIGHT - MARGIN):
            self.canvas.focus_set()
            
            # get row and col numbers from x,y coordinates
            row, col = (y - MARGIN) / SIDE, (x - MARGIN) / SIDE
            
            # if cell was selected already - deselect it
            if (row, col) == (self.row, self.col):
                self.row, self.col = -1, -1
            else:
                self.row, self.col = row, col
        else:
            self.row, self.col = -1, -1
        self.__draw_cursor()

    def __key_pressed(self, event):
        if self.row >= 0 and self.col >= 0 and event.char in "1234567890":
            try:
                self.game.puzzle[self.row][self.col] = int(event.char)
                self.col, self.row = -1, -1
                self.__draw_puzzle()
                self.__draw_cursor()
            except (RuntimeError, TypeError, ValueError):
                pass

    def __clear_answers(self):
        self.game.start()
        self.__draw_puzzle()
        
    
    def __solve_sudoku(self):
         self.unsolved_puzzle = copy.deepcopy(self.game.puzzle)
         backtracking_solver = BacktrackingSudokuSolver(self.game.puzzle)
         backtracking_solver.solve()
         self.__draw_puzzle(backtracking_solver.sudoku_matrix)
         
    def determine_solved_numbers(self,row,column):
        print (self.unsolved_puzzle[row][column])
        if (self.unsolved_puzzle[row][column] == 0):
            return "red"
        return "black"

class SudokuGame(object):
    def __init__(self):
        self.start_puzzle = [[0]*9]*9

    def start(self):
        self.puzzle = []
        for i in xrange(9):
            self.puzzle.append([])
            for j in xrange(9):
                self.puzzle[i].append(self.start_puzzle[i][j])
        


if __name__ == '__main__':
    network.learn()
    game = SudokuGame()
    game.start()

    root = Tk()
    gui = SudokuUI(root, game)
    root.geometry("%dx%d" % (WIDTH, HEIGHT + 40))
    root.mainloop()
