# Course: CS 261
# Author: Sam Silber
# Assignment: 6
# Description: Undirected Graph implementation

import heapq
from collections import deque


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Add new vertex to the graph. If vertex with
        the same name is already present in the graph, the method does nothing
        """

        # If the vertex already exists, do nothing
        if v in self.adj_list:
            return

        # Add the provided value as a new key in the list
        self.adj_list[v] = []

    def add_edge(self, u: str, v: str) -> None:
        """
        Adds a new edge to the graph, connecting two vertices with provided names. If
        either (or both) vertex names do not exist in the graph, this method will first create them
        and then create an edge between them. If an edge already exists in the graph, or if u and v
        refer to the same vertex, the method does nothing (no exception needs to be raised).
        """

        # If u and v make the same reference, do nothing
        if u == v:
            return

        # If u is not a vertex, add it
        if u not in self.adj_list:
            self.add_vertex(u)

        # If v is not a vertex, add it
        if v not in self.adj_list:
            self.add_vertex(v)

        # If u and v are already connected, do nothing
        if u in self.adj_list[v] and v in self.adj_list[u]:
            return

        # Otherwise, make the connection
        self.adj_list[v].append(u)
        self.adj_list[u].append(v)

    def remove_edge(self, v: str, u: str) -> None:
        """
        Removes an edge between two vertices with provided names. If either (or
        both) vertex names do not exist in the graph, or if there is no edge between them, the
        method does nothing (no exception needs to be raised).
        """

        # Do nothing if either of the vertices aren't in the list
        if u not in self.adj_list or v not in self.adj_list:
            return

        # Do nothing if there is no edge between the two verticies
        if u not in self.adj_list[v] and v not in self.adj_list[u]:
            return

        # Otherwise, remove the connection
        self.adj_list[v].remove(u)
        self.adj_list[u].remove(v)

    def remove_vertex(self, v: str) -> None:
        """
        Removes a vertex with a given name and all edges incident to it from the
        graph. If the given vertex does not exist, the method does nothing (no exception needs to
        be raised).
        """

        # If the vertex doesn't exist, do nothing
        if v not in self.adj_list:
            return

        # Otherwise, entirely remove the vertex
        self.adj_list.pop(v, None)
        for vertex in self.adj_list:
            if v in self.adj_list[vertex]:
                self.adj_list[vertex].remove(v)

    def get_vertices(self) -> []:
        """
        Return list of vertices in the graph (any order)
        """

        return [v for v in self.adj_list]

    def get_edges(self) -> []:
        """
        Return list of edges in the graph (any order)
        """
        out = []
        for v in self.adj_list:
            for e in self.adj_list[v]:
                if (e, v) not in out:
                    out.append((v, e))
        return out

    def is_valid_path(self, path: []) -> bool:
        """
        Takes a list of vertex names and returns True if the sequence of vertices
        represents a valid path in the graph. Empty path is considered valid.
        """

        # Empty path is valid
        if not path:
            return True

        # Get the list of edges and set pointers for the current/previous vertex being assessed
        edges = self.get_edges()
        cur_v = None
        prev_v = None

        for i, p in enumerate(path):

            cur_v = p

            # If it's the first vertex, check if the vertex exists
            if i == 0:
                if p not in self.adj_list:
                    return False

            # For all other vertices, check if the current vertex/previous vertex combination
            # is in the list of edges
            else:
                if (cur_v, prev_v) not in edges and (prev_v, cur_v) not in edges:
                    return False

            # Set the previous vertex to the current vertex for the next iteration
            prev_v = cur_v

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during DFS search
        """

        # Return empty list if starting vertex is not in the graph
        if v_start not in self.adj_list:
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

            # Check the connected vertices and add them to the stack in reverse alphabetical order
            # so the top of the stack is in ascending lexicographical order
            for v in sorted(self.adj_list[cur_v], reverse=True):
                if v not in order:
                    stack.append(v)

        return order

    def bfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during BFS search
        """

        # Return empty list if starting vertex is not in the graph
        if v_start not in self.adj_list:
            return []

        # Create a queue for the search and an empty list for returning the order
        queue = deque()
        queue.appendleft(v_start)

        order = []

        # While the queue isn't empty
        while queue:

            # Dequeue the current vertex
            cur_v = queue.pop()

            # If it's not already in the return list, add it
            if cur_v not in order:
                order.append(cur_v)

                # If the value just added is the optional end parameter, end the loop
                if cur_v == v_end:
                    break

            # Check the connected vertices and add them to the queue in alphabetical order
            # so they are assessed in ascending lexicographical order
            for v in sorted(self.adj_list[cur_v], reverse=False):
                if v not in order:
                    queue.appendleft(v)

        return order

    def count_connected_components(self):
        """
        Return number of connected componets in the graph
        """

        # Create a counter for the number of connected components and a tracker for visited vertices
        count = 0
        visited = []

        # Check each vertex
        for v in self.adj_list:

            # If the vertex hasn't been visited:
            if v not in visited:

                # Add it to the visited list
                visited.append(v)

                # Get the dfs of the vertex to get all of its connected elements
                dfs = self.dfs(v)

                # Add each item from the dfs to visited; all of these vertices are of the same connected component
                for i in dfs:
                    visited.append(i)

                # Increment the counter
                count += 1

        return count

    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """
        # Run the process for each edge in the case of disjoint graphs
        for e in range(len(self.get_edges())):

            # Get the first vertex in the tuple
            v_start = self.get_edges()[e][0]

            # Create a stack for currently visited edge and an empty list for visited vertices
            stack = [(None, v_start)]
            visited = []

            # While the stack isn't empty
            while stack:

                # Pop the top edge of the stack and get the previously visited vertex and current vertex
                pop = stack.pop()
                prev_v = pop[0]
                cur_v = pop[1]

                # For each neighbor of the current vertex
                for v in sorted(self.adj_list[cur_v], reverse=True):

                    # Pass if it's checking the previously visited vertex
                    if v == prev_v:
                        pass
                    # If the vertex has been visited, there's a cycle
                    elif v in visited:
                        return True

                    # Otherwise, add the vertex to visited and add the edge to the stack
                    else:
                        visited.append(cur_v)
                        stack.append((cur_v, v))

        return False


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)

    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)

    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')

    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())


