import numpy as np
import random
from collections import deque
import torch
from tqdm import tqdm
from typing import Optional


class MapGenerator:
    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        num_maps: int = 10,
        sparsity_low: float = 53,
        sparsity_high: float = 88,
        max_path_len: int = 13,
        exclude_maps: Optional[dict] = None,
        wall_coords: Optional[list] = None,
        space_coords: Optional[list] = None,
    ):
        """
        wall_coords: list of tuples, each tuple is a coordinate (x, y) to required to be a wall
        """
        self.width = width
        self.height = height
        self.num_maps = num_maps
        self.max_path_len = max_path_len
        self.sparsity_low = sparsity_low
        self.sparsity_high = sparsity_high
        self.wall_coords = wall_coords
        self.space_coords = space_coords

        if exclude_maps is not None:
            self.exclude_maps = exclude_maps
        else:
            self.exclude_maps = {}

    def _bfs_longest_path(self, grid, start_row, start_col):
        rows, cols = grid.shape
        visited = np.full((rows, cols), False)
        queue = deque([(start_row, start_col, 0)])  # (row, col, distance)
        visited[start_row, start_col] = True
        max_distance = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        while queue:
            r, c, dist = queue.popleft()
            max_distance = max(max_distance, dist)

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and grid[nr, nc] == "O"
                    and not visited[nr, nc]
                ):
                    visited[nr, nc] = True
                    queue.append((nr, nc, dist + 1))

        return max_distance

    def _find_longest_connected_distance(self, grid):
        rows, cols = grid.shape
        max_distance = 0

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == "O":
                    # Perform BFS from this 'O' and find the longest path
                    distance = self._bfs_longest_path(grid, r, c)
                    max_distance = max(max_distance, distance)

        return max_distance

    # using random approach
    def _initialize_grid(
        self, width, height, border_fill_prob=0.5, interior_fill_prob=0.5
    ):
        """
        Initialize the grid with random walls and open spaces.
        :param width: The width of the grid.
        :param height: The height of the grid.
        :param border_fill_prob: Probability of a border cell being open space.
        :param interior_fill_prob: Probability of an interior cell being open space.
        :return: Initialized grid.
        """
        grid = np.full(
            (height, width), "#", dtype=str
        )  # Initialize all cells with walls

        for r in range(height):
            for c in range(width):
                if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                    # Border cells
                    grid[r, c] = "O" if random.random() < border_fill_prob else "#"
                else:
                    # Interior cells
                    grid[r, c] = "O" if random.random() < interior_fill_prob else "#"

        return grid

    def _is_connected(self, grid):
        """
        Check if all 'O' cells in the grid are connected.
        :param grid: 2D array of '#' and 'O'.
        :return: True if all 'O' cells are connected, False otherwise.
        """
        width, height = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        def bfs(start_x, start_y):
            queue = deque([(start_x, start_y)])
            visited[start_x, start_y] = True
            count = 1  # Start with the initial cell
            while queue:
                x, y = queue.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and not visited[nx, ny]
                        and grid[nx, ny] == "O"
                    ):
                        visited[nx, ny] = True
                        queue.append((nx, ny))
                        count += 1
            return count

        # Find the first 'O'
        for x in range(width):
            for y in range(height):
                if grid[x, y] == "O":
                    # Perform BFS/DFS from the first 'O'
                    connected_count = bfs(x, y)
                    total_o_count = np.sum(grid == "O")
                    return connected_count == total_o_count
        return True  # If there are no 'O's at all

    def _open_space_to_wall(self, grid, N):
        # Create a copy of the grid to apply changes without affecting the original during the process
        new_grid = np.copy(grid)

        rows, cols = grid.shape

        # Directions for the 8 neighboring cells, including diagonals
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == "O":  # Only consider open spaces
                    # Count the number of open spaces in the neighboring cells
                    open_space_count = 0
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == "O":
                            open_space_count += 1

                    # If the number of open spaces exceeds N, turn this space into a wall
                    if open_space_count > N:
                        new_grid[r, c] = "#"

        return new_grid

    def _wall_to_open_space(self, grid, M):
        # Create a copy of the grid to apply changes without affecting the original during the process
        new_grid = np.copy(grid)

        rows, cols = grid.shape

        # Directions for the 8 neighboring cells, including diagonals
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == "#":  # Only consider walls
                    # Count the number of walls in the neighboring cells
                    wall_count = 0
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == "#":
                            wall_count += 1

                    # If the number of walls equals or exceeds M, turn this wall into an open space
                    if wall_count > M:
                        new_grid[r, c] = "O"

        return new_grid

    def _apply_cellular_automata(self, grid, N=6, M=4):
        grid = self._open_space_to_wall(grid, N=N)
        # grid = self._wall_to_open_space(grid, M=M)
        return grid

    def _generate_map(self, width, height, iterations=2):
        """
        Generate a randomized map with connected open spaces.
        :param width: The width of the grid.
        :param height: The height of the grid.
        :param iterations: Number of iterations for the smoothing process.
        :return: Generated map grid.
        """
        grid = self._initialize_grid(width, height)

        for i in range(iterations):
            grid = self._apply_cellular_automata(grid)

        return grid

    def _calculate_o_percentage(self, grid):
        """
        Calculate the percentage of 'O' cells in the grid.
        :param grid: 2D numpy array of '#' and 'O'.
        :return: Percentage of 'O' cells in the grid.
        """
        # Convert grid to numpy array if it's not already
        grid = np.array(grid)

        # Count the number of 'O' cells
        num_o_cells = np.sum(grid == "O")

        # Calculate the total number of cells
        total_cells = grid.size

        # Calculate the percentage of 'O' cells
        o_percentage = (num_o_cells / total_cells) * 100

        return o_percentage

    def _add_walls(self, grid):
        border_char = "#"

        # Get the original array dimensions
        rows, cols = grid.shape

        # Create a new array with an extra row and column for the borders
        new_array = np.full((rows + 2, cols + 2), border_char, dtype=grid.dtype)

        # Place the original array in the center of the new array
        new_array[1:-1, 1:-1] = grid

        return new_array

    def print_grid(self, grid):
        """
        Print the grid.
        :param grid: The grid (2D array) to print.
        """
        for row in grid:
            print("".join(row))

    def print_grid_from_key(self, key):
        rows = key.split("\\")
        for row in rows:
            print(row)
        print("\n")

    def _generate_key(self, grid):
        key = ""
        for i, row in enumerate(grid):
            key += "".join(row)
            if i < len(grid) - 1:
                key += "\\"
        return key

    def _pass_wall_constraint(self, grid):
        if not self.wall_coords:
            return True

        for coord in self.wall_coords:
            if grid[coord[0], coord[1]] == "O":
                return False

        return True

    def _pass_space_constraint(self, grid):
        if not self.space_coords:
            return True

        for coord in self.space_coords:
            if grid[coord[0], coord[1]] == "#":
                return False

        return True

    def generate_diverse_maps(self):
        map_dict = {}

        print(f"Generating map layouts")
        for i in tqdm(range(self.num_maps)):
            while True:
                map_grid = self._generate_map(self.width - 2, self.height - 2)

                sparsity = self._calculate_o_percentage(map_grid)

                if not self._pass_wall_constraint(map_grid):
                    continue

                if not self._pass_space_constraint(map_grid):
                    continue

                if not (
                    self.sparsity_low <= sparsity and sparsity <= self.sparsity_high
                ):
                    continue

                map_grid = self._add_walls(map_grid)

                key = self._generate_key(map_grid)

                if key in self.exclude_maps or key in map_dict:
                    continue

                if not (self._is_connected(map_grid)):
                    continue

                longest_dist = self._find_longest_connected_distance(map_grid)
                if longest_dist >= self.max_path_len:
                    continue

                print(sparsity)
                print(longest_dist)
                self.print_grid_from_key(key)
                map_dict[key] = True
                break

        map_dict = {i: key for i, key in enumerate(map_dict.keys())}
        return map_dict


def main():
    map_generator = MapGenerator()
    maps = map_generator.generate_diverse_maps()

    # just to visualize
    # for i in range(10):
    #     map_generator.print_grid_from_key(maps[i])
    #     print('\n')

    # print(maps.keys())

    # map_generator = RecursiveMapGenerator()
    # for i in range(10):
    #     layout = map_generator.generate_maze()
    #     map_generator.print_maze(layout)
    #     print('\n')


if __name__ == "__main__":
    main()
