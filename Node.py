#klasa koja reprezenutuje jedan cvor u doubly linked listi
class Node(object):

    #svaki cvor ima informaciju ko je levo,desno,gore i dole u odnosu na njega
    #pored prethodnih sadrzi i informaciju o tome kojoj koloni priprada..pokazivac na column_node
    def __init__(self):
        self.up = None
        self.down = None
        self.right = None
        self.left = None
        self.column_node = None
        self.row = -1
        self.column = -1