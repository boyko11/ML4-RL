

def construct_grid_and_terminals(num_rows=100, num_columns=100):

    no_obstacles_row = [-0.04] * num_columns
    top_row = no_obstacles_row.copy()
    top_row[-1] = 1

    obstacles_row = [-0.04] * num_columns
    for obstacle_index in range(1, len(obstacles_row), 2):
        obstacles_row[obstacle_index] = None

    top_row_minus_1 = obstacles_row.copy()
    top_row_minus_1[-1] = -1

    grid = [no_obstacles_row] * num_rows

    for obstacles_row_index in range(3, len(grid), 2):
        grid[obstacles_row_index] = obstacles_row

    grid[0] = top_row
    grid[1] = top_row_minus_1

    terminal_plus_1 = (num_columns - 1, num_rows - 1)
    terminal_minus_1 = (num_columns - 1, num_rows - 2)
    terminals = [terminal_plus_1, terminal_minus_1]

    return grid, terminals

