from Node import Node


class ColumnNode(Node):

    def __init__(self,name):
        Node.__init__(self)
        self.size = 0
        self.name = name

