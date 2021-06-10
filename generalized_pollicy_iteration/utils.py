from colorama import Fore


def print_v(v, columns: int):
    """
    Print Grid
    -----------
    """
    grid = ""
    for i, v in enumerate(v):
        grid = grid + f"{v:.2f}" + (f'\n' if (i + 1) % columns == 0 else '\t')

    print(grid)


def print_policy(maze, policy, rows, columns):
    actions = [u"\u2191", u"\u2193", u"\u2190", u"\u2192"]

    for i in range(rows):

        for j in range(columns):
            if maze[i][j] == 'u':
                print(Fore.WHITE + str(maze[i][j]), end=" ")
            elif maze[i][j] == 'c':
                state_probs = policy[i * columns + j]
                if i == rows-1:
                    print(Fore.GREEN + 'o', end=" ")
                elif i == 0:
                    print(Fore.GREEN + 'i', end=" ")
                else:
                    print(Fore.GREEN + ''.join([actions[i] for i, value in enumerate(state_probs) if value != 0]), end=" ")
            else:
                print(Fore.RED + str(maze[i][j]), end=" ")

        print('\n')

