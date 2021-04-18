# Course: CS261 - Data Structures
# Author: Sam Silber
# Assignment: CS 261
# Description: Directed Graph implementation

from heapq import heappush, heappop
from collections import deque


class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        Adds a new vertex to the graph. This method returns a single integer -
        the number of vertices in the graph after the addition.
        """

        # Increase the vertex count, and then make a list of lists depending on the count size
        self.v_count += 1
        self.adj_matrix = [[0] * self.v_count for i in range(self.v_count)]

        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Adds a new edge to the graph. If either (or both) vertex indices do not exist in the graph,
        or if the weight is not a positive integer, or if src and dst refer to the same vertex,
        the method does nothing. If an edge already exists in the graph, the method will update its weight.
        """
        rows = len(self.adj_matrix)
        cols = len(self.adj_matrix[0])

        # Do nothing for a whole myriad of reasons
        if src < 0 or src >= rows or src >= cols:
            return None
        if dst < 0 or dst >= rows or dst >= cols:
            return None
        if weight < 0:
            return None
        if src == dst:
            return None

        # Update the correct index of the matrix with the new weight
        self.adj_matrix[src][dst] = weight
        return None

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Removes an edge between two vertices with provided indices. If either (or
        both) vertex indices do not exist in the graph, or if there is no edge between them, the
        method does nothing
        """
        rows = len(self.adj_matrix)
        cols = len(self.adj_matrix[0])

        # Do nothing for the reasons listed above
        if src < 0 or src >= rows or src >= cols:
            return
        if dst < 0 or dst >= rows or dst >= cols:
            return
        if self.adj_matrix[src][dst] == 0:
            return

        # update the weight to 0 at the given coordinate
        self.adj_matrix[src][dst] = 0
        return None

    def get_vertices(self) -> []:
        """
        Returns a list of vertices of the graph. Order of the vertices in the list does not
        matter
        """

        # return a list with 1 record for each row
        return [x for x in range(len(self.adj_matrix))]

    def get_edges(self) -> []:
        """
        Returns a list of edges in the graph. Each edge is returned as a tuple of two
        incident vertex indices and weight. First element in the tuple refers to the source vertex.
        Second element in the tuple refers to the destination vertex. Third element in the tuple is
        the weight of the edge. Order of the edges in the list does not matter
        """
        out = []

        for i, row in enumerate(self.adj_matrix):
            for j, col in enumerate(row):
                if self.adj_matrix[i][j] != 0:
                    out.append((i, j, self.adj_matrix[i][j]))

        return out

    def is_valid_path(self, path: []) -> bool:
        """
        Takes a list of vertex indices and returns True if the sequence of vertices
        represents a valid path in the graph. Empty path is considered valid.
        """
        # Empty path is valid
        if not path:
            return True

        # Get the list of edges and get the vertex/edge from each
        edges = self.get_edges()
        tuples = [(e[0], e[1]) for e in edges]

        # Look if each sequential pair of indices provided is in the list of tuples created above
        for i in range(len(path) - 1):
            cur_v = path[i]
            next_v = path[i + 1]

            search = (cur_v, next_v)

            if search not in tuples:
                return False
        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during DFS search
        """

        # Get the number of rows and columns in the matrix
        rows = len(self.adj_matrix)
        cols = len(self.adj_matrix[0])

        # Return empty list if start_v is out of index
        if v_start < 0 or v_start >= rows or v_start >= cols:
            return []

        # Create a stack for the search and an empty list for returning the order
        stack = [v_start]
        order = []

        # While the stack isn't empty
        while stack:

            # Pop the top element of the stack
            cur_v = stack.pop()

            # If it's not already in the return list, add it
            if cur_v not in order:
                order.append(cur_v)

                # If the value just added is the optional end parameter, end the loop
                if cur_v == v_end:
                    break

            # Check the connected vertices in the row and add them to their own list
            row = []
            for i, v in enumerate(self.adj_matrix[cur_v]):
                if v != 0:
                    if i not in order:
                        row.append(i)

            # add the vertices in the row to the stack in reverse alphabetical order
            # so the top of the stack is in ascending lexicographical order
            for v in sorted(row, reverse=True):
                stack.append(v)

        return order

    def bfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during BFS search
        """

        # Get the number of rows and columns in the matrix
        rows = len(self.adj_matrix)
        cols = len(self.adj_matrix[0])

        # Return empty list if start_v is out of index
        if v_start < 0 or v_start >= rows or v_start >= cols:
            return []

        # Create a queue for the search and an empty list for returning the order
        queue = deque()
        queue.appendleft(v_start)

        order = []

        # While the stack isn't empty
        while queue:

            # Dequeue the current vertex
            cur_v = queue.pop()

            # If it's not already in the return list, add it
            if cur_v not in order:
                order.append(cur_v)

                # If the value just added is the optional end parameter, end the loop
                if cur_v == v_end:
                    break

            # Check the connected vertices in the row and add them to their own list
            row = []
            for i, v in enumerate(self.adj_matrix[cur_v]):
                if v != 0:
                    if i not in order:
                        row.append(i)

            # add the vertices in the row to the queue in alphabetical order
            # so they are assessed in ascending lexicographical order
            for v in sorted(row, reverse=False):
                queue.appendleft(v)

        return order

    def has_cycle(self):
        """
        Returns True if there is at least one cycle in the graph. If the graph is acyclic,
        the method returns False.
        Used: https://www.baeldung.com/cs/detecting-cycles-in-directed-graph
        """

        # Create a dict of each vertex flagging them as "Not Visited"
        # Also create a list with a bool for whether or not there's a cycle that can be passed recursibely
        vertices = self.get_vertices()
        visited = {}
        cycle_exists = [False]

        for v in vertices:
            visited[v] = "Not Visited"

        # Go through all verticies
        for v in vertices:

            # If the vertex hasn't been visited, add it to the stack and do a DFS
            if visited[v] == "Not Visited":
                stack = []
                stack.append(v)
                self.dfs_process(stack, visited, cycle_exists)

            # If cycle_exists == True, break out and return
            if cycle_exists[0]:
                break

        return cycle_exists[0]

    def dfs_process(self, stack, visited, cycle_exists):
        """
        Helper function for has_cycle-- does a DFS on each adjacent vertex
        """

        # For each adjacent vertex
        for i, v in enumerate(self.adj_matrix[stack[-1]]):

            # If the weight > 0 (i.e., the edge exists
            if v != 0:

                # If the adjacent vertex is "In Stack", it has already been visited, and there is a cycle
                if visited[i] == "In Stack":
                    cycle_exists[0] = True
                    return

                # Otherwise, add the adjacent vertex to the stack and keep on dfsing
                elif visited[i] == "Not Visited":
                    stack.append(i)
                    visited[i] = "In Stack"
                    self.dfs_process(stack, visited, cycle_exists)

        # Move the vertex to the "Done" group as it has been fully searched
        visited[stack[-1]] = "Done"
        stack.pop()

    def dijkstra(self, src: int) -> []:
        """
        Implements the Dijkstra algorithm to compute the length of the shortest path
        from a given vertex to all other vertices in the graph. It returns a list with one value per
        each vertex in the graph, where value at index 0 is the length of the shortest path from
        vertex SRC to vertex 0, value at index 1 is the length of the shortest path from vertex SRC
        to vertex 1 etc. If a certain vertex is not reachable from SRC, returned value should be
        INFINITY (in Python, use float(‘inf’)).
        Source: follows algo laid out in the modules
        """

        # Create an empty dict for visited vertices
        visited = {}

        # Initialize an empty priority queue and add tuple with src and a distance of 0
        heap = []
        heappush(heap, (0, src))

        # While the heap isn't empty
        while heap:

            # Get the first element and get the vertex and distance
            tup = heappop(heap)
            v = tup[1]
            d = tup[0]

            # If the vertex is not in the visited dictionary
            if v not in visited:

                # Add it to the visited dict
                visited[v] = d

                # For each adjacent vertex
                for ii, vi in enumerate(self.adj_matrix[v]):

                    # If the weight > 0 (i.e., the edge exist
                    if vi != 0:
                        # Get the distance associated with the edge and add it and its vertex to the heap
                        di = vi
                        heappush(heap, (di + d, ii))



        # Create a list for output the length of the number of vertices
        vertices = self.get_vertices()
        out = [float('inf')] * len(vertices)

        # Go through the visited dict and put the appropriate distances at each point in the output list
        for vertex in vertices:
            if vertex in visited:
                out[vertex] = visited[vertex]

        return out


if __name__ == '__main__':

    # print("\nPDF - method add_vertex() / add_edge example 1")
    # print("----------------------------------------------")
    # g = DirectedGraph()
    # print(g)
    # for _ in range(5):
    #     g.add_vertex()
    # print(g)
    #
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # for src, dst, weight in edges:
    #     g.add_edge(src, dst, weight)
    # print(g)
    #
    # print("\nPDF - method get_edges() example 1")
    # print("----------------------------------")
    # g = DirectedGraph()
    # print(g.get_edges(), g.get_vertices(), sep='\n')
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # print(g.get_edges(), g.get_vertices(), sep='\n')
    #
    # print("\nPDF - method is_valid_path() example 1")
    # print("--------------------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    # for path in test_cases:
    #     print(path, g.is_valid_path(path))
    #
    # print("\nPDF - method dfs() and bfs() example 1")
    # print("--------------------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # for start in range(5):
    #     print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')
    #
    # print("\nPDF - method has_cycle() example 1")
    # print("----------------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    #
    # edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    # for src, dst in edges_to_remove:
    #     g.remove_edge(src, dst)
    #     print(g.get_edges(), g.has_cycle(), sep='\n')
    #
    # edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    # for src, dst in edges_to_add:
    #     g.add_edge(src, dst)
    #     print(g.get_edges(), g.has_cycle(), sep='\n')
    # print('\n', g)
    # #
    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
