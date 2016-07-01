from ColumnNode import ColumnNode
import numpy as np
import sys

from Node import Node


class DancingLinksX(object):

    def __init__(self,sudoku_matrix):
        self.root = None
        self.solutions = []
        self.sparse_matrix = None
        self.sudoku_matrix = sudoku_matrix
        self.create_sparse_matrx()
        #self.print_sparse_matrix()
        self.create_double_linked_list()

    def cover(self,column):
        column.right.left = column.left
        column.left.right = column.right

        # 'brisanje' redova koji su u preseku sa tom kolonom
        row = column.down
        while row != column:
            right_neighbour = row.right
            while right_neighbour != row:
                right_neighbour.down.up = right_neighbour.up
                right_neighbour.up.down = right_neighbour.down
                right_neighbour.column_node.size  -= 1
                right_neighbour = right_neighbour.right
            row = row.down

    def uncover(self,column):
        row = column.up
        while row != column:
            left_neighbour = row.left
            while left_neighbour != row:
                left_neighbour.column_node.size += 1
                left_neighbour.down.up = left_neighbour
                left_neighbour.up.down = left_neighbour
                left_neighbour = left_neighbour.left
            row = row.up
        column.right.left = column
        column.left.right = column

    #k - parameter dubine
    def search(self,k):

        if self.root.right == self.root:
            return self.solutions
        else:
            column = self.choose_next_column()
            self.cover(column)
            row = column.down
            while row != column:
                self.solutions.append(row)
                right_neighbour = row.right
                while right_neighbour != row:
                    self.cover(right_neighbour.column_node)
                    right_neighbour = right_neighbour.right
                success = self.search(k+1)
                if success:
                    return success
                row = self.solutions.pop()
                column = row.column_node
                left_neighbour = row.left
                while left_neighbour != row:
                    self.uncover(left_neighbour.column_node)
                    left_neighbour = left_neighbour.left
                row = row.down
            self.uncover(column)
            return None

    #sparse matrica - matrica 0 i 1
    #729 redova - svaka celija u sudoku tabeli moze sadrzati bilo koji broj, prema tome,broj redova je 9^3 (81 * 9)
    #324 kolone - svaka celija ima 4 'zabrane':
    #I - samo jedan broj u jednoj celiji
    #II - jedinstven broj u redu
    #III - jedinstven broj u koloni
    #IV - jedinstven broj u regionu (maloj podmatrici)
    def create_sparse_matrx(self):
        self.sparse_matrix =  []

        for row in xrange(9):
            for column in xrange(9):
                digit = self.sudoku_matrix[row][column]
                if digit in xrange(1,10):
                    box = (row - (row % 3)) + (column / 3)
                    one_row = self.one_row_sparse_matrix(row,column,box,digit)
                    self.sparse_matrix.append(one_row)
                else:
                    for d in xrange(1,10):
                        box = (row - (row % 3)) + (column / 3)
                        one_row = self.one_row_sparse_matrix(row,column,box,d)
                        self.sparse_matrix.append(one_row)




    def one_row_sparse_matrix(self,row,column,box,value):
        one_row = [0] * 324
        one_row[row * 9 + column] = 1
        one_row[81 + (row * 9) + (value - 1)] = 1
        one_row[162 + (column * 9) + (value - 1)] = 1
        one_row[243 + (box * 9) + (value - 1)] = 1
        return one_row


    def create_double_linked_list(self):

        #kreiranje root-a
        #preko njega se pristupa svim ostalim cvorovima liste
        self.root = ColumnNode("root")

        #kreiranje kolona
        columns = []
        for i in xrange(len(self.sparse_matrix[0])):
            column = ColumnNode(i)
            columns.append(column)

        #povezivanje kolona
        first_column = None
        previous_column = None

        for column in columns:
            if first_column is None:
                first_column = column
            elif previous_column is not None:
                column.left = previous_column
                previous_column.right = column
            previous_column = column

        #povezivanje poslednjeg,prvog cvora i korena
        first_column.left = self.root
        self.root.right = first_column
        previous_column.right = self.root
        self.root.left = previous_column



        columns_rows = {}

        for row in xrange(len(self.sparse_matrix)):

            first_node_row = None
            previous_node_row = None

            # kreiranje jednog reda cvorova i postavljanje levih i desnih suseda
            for c in xrange(len(self.sparse_matrix[row])):

                if self.sparse_matrix[row][c] == 1:

                    row_node = Node()
                    row_node.row = row
                    row_node.column = c
                    columns_rows[(row,c)] = row_node
                    if first_node_row is None:
                        first_node_row = row_node
                    elif previous_node_row is not None:
                        previous_node_row.right = row_node
                        row_node.left = previous_node_row
                    previous_node_row = row_node

            first_node_row.left = previous_node_row
            previous_node_row.right = first_node_row


        #povezivanje cvorova u jednoj koloni - postavljanje up i down suseda
        current_column = self.root.right
        while current_column != self.root:
                rows_from_column = self.get_rows_from_column(columns_rows, current_column.name)
                current_column.size = len(rows_from_column)

                first_row_column_node = None
                previous_row_column_node = None
                for r in rows_from_column:
                    node = columns_rows[(r, current_column.name)]
                    node.column_node = current_column
                    if first_row_column_node is None:
                        first_row_column_node = node
                    elif previous_row_column_node is not None:
                        node.up = previous_row_column_node
                        previous_row_column_node.down = node
                    previous_row_column_node = node


                current_column.up = previous_row_column_node
                previous_row_column_node.down = current_column

                current_column.down = first_row_column_node
                first_row_column_node.up = current_column


                current_column = current_column.right

    def choose_next_column(self):

        best_column = None
        best_size = sys.maxint

        current_column = self.root.right

        while current_column != self.root:
            if current_column.size < best_size:
                best_size = current_column.size
                best_column = current_column
            current_column = current_column.right

        return best_column


    def get_rows_from_column(self,columns_and_nodes,c):

        column_node_keys = columns_and_nodes.keys()
        rows = [row_index for (row_index, col_index) in  column_node_keys if col_index == c]
        rows == list(set(rows))
        rows.sort()
        return rows


    def print_sparse_matrix(self):
        print(len(self.sparse_matrix))
        for r in xrange(len(self.sparse_matrix)):
            print "Broj kolona: " + str(len(self.sparse_matrix[r]))






















