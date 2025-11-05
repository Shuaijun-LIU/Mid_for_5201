from maze3.maze import Maze
from maze3.util import Coordinates
from random import choice


class KruskalMazeGenerator():
    """
    Kruskal's algorithm maze generator.
    """

    def generateMaze(self, maze: Maze):
        # Step 1: Initialize edge collection and sort by weight
        all_edges = []  # List to store all edges

        # Traverse each cell in the maze
        for row_idx in range(maze.rowNum()):
            for col_idx in range(maze.colNum()):

                current_coord = Coordinates(row_idx, col_idx)  # Get current cell coordinates
                neighboring_coords = maze.neighbours(current_coord)  # Get adjacent cell coordinates

                # Create edges and add to all_edges list
                for neighbor_coord in neighboring_coords:  # For each adjacent cell
                    # Add each edge only once

                    # Add edge only once to avoid duplicates, using lexicographic order for uniqueness
                    if (current_coord.getRow(), current_coord.getCol()) < (neighbor_coord.getRow(), neighbor_coord.getCol()):
                        # Always add edge from cell with smaller (row, col) to cell with larger (row, col)
                        # (current_coord.getRow(), current_coord.getCol()) represents current cell coordinates as a tuple
                        # (neighbor_coord.getRow(), neighbor_coord.getCol()) represents adjacent cell coordinates
                        # The < operator compares tuples element by element (first row, then col)
                        weight_difference = abs(current_coord.getWeight() - neighbor_coord.getWeight())  # Calculate weight difference for sorting
                        # Weight difference measures the "distance" or "connection cost" between two cells

                        all_edges.append((weight_difference, current_coord, neighbor_coord))  # Add edge to list with weight and two coordinates
                        # weight_difference: edge weight difference for sorting
                        # current_coord: current cell coordinates
                        # neighbor_coord: adjacent cell coordinates

        # Sort edges by weight (Kruskal algorithm starts from minimum weight edges)
        sorted_edges = sorted(all_edges, key=lambda edge: edge[0])

        # Step 2: Initialize union-find data structure to track cell sets
        parent_map = {}  # Store parent node for each cell
        rank_map = {}  # Store rank (depth) of each cell's set

        # Initialize each cell so its parent points to itself (each cell starts as independent set)
        for coord in maze.getCoords():

            parent_map[coord] = coord  # Each cell initially points to itself
            rank_map[coord] = 0  # Each cell initially has rank 0

        # Step 3: Define helper functions for union-find operations
        # Find root node helps decide whether to remove walls:
        # If two cells have different root nodes, they belong to different sets,
        # so we can safely remove the wall to connect them (no cycle will form).
        # If root nodes are the same, they are already connected,
        # so we don't need to remove the wall, avoiding cycles.
        # All elements in a set initially point to themselves as "parent nodes".
        # When sets merge, some elements point to others, forming a "tree" structure.
        # Root node is the top element that all elements eventually point to.
        # Path compression optimization reduces tree depth to avoid deep nesting and speed up lookups.
        def find_parent(coordinate):

            if parent_map[coordinate] != coordinate:
                # This checks if current cell coordinate is the root of its own set.
                # If parent_map[coordinate] == coordinate, then coordinate itself is the root.

                parent_map[coordinate] = find_parent(parent_map[coordinate])
                # If coordinate is not the root, recursively call find_parent(parent_map[coordinate])
                # to continue finding the root of coordinate's parent node

            return parent_map[coordinate]  # Finally return the root node of coordinate, i.e., parent_map[coordinate]


        def union_cells(coord1, coord2):  # Union operation: merge two sets by rank
            # Union by rank effectively reduces tree height, improving lookup and merge efficiency
            root1 = find_parent(coord1)
            root2 = find_parent(coord2)

            if root1 != root2:  # Only merge if they belong to different sets

                if rank_map[root1] > rank_map[root2]:  # Attach smaller tree under larger tree based on rank
                    parent_map[root2] = root1

                elif rank_map[root1] < rank_map[root2]:
                    parent_map[root1] = root2

                else:
                    parent_map[root2] = root1  # If ranks are equal, arbitrarily choose root1 as parent
                    rank_map[root1] += 1  # Increase root1's rank

        # Step 4: Process edges - execute Kruskal algorithm to generate maze
        # Traverse all edges and decide whether to remove walls to build maze structure
        for edge_info in sorted_edges:  # Each iteration takes one edge edge_info (contains weight and two endpoints coord1 and coord2)
            weight, coord1, coord2 = edge_info  # Check if two cells belong to different sets (to avoid cycles)

            if find_parent(coord1) != find_parent(coord2):
                # Remove wall between coord1 and coord2
                maze.removeWall(coord1, coord2)  # If two cells are not in the same set, remove wall between them to connect them

                union_cells(coord1, coord2)  # Merge the sets containing coord1 and coord2, indicating they are now connected.
