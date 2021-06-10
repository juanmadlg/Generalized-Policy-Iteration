class MazeEnvironment:
    def __init__(self, maze: list):
        '''
        Initialization of the Maze Environment
        -------------
        maze : list
          List of list of c (clear) or w (wall) that represent the maze
        '''
        self.maze = maze
        self.num_rows = len(maze)
        self.num_columns = len(maze[0])

        # Available actions
        self.actions = ['up', 'down', 'left', 'right']

        # Status
        self.current = None
        self.previous = None

        self.terminal_state = (self.num_rows - 1, self.maze[-1].index('c'))

    def reset(self):
        '''
        Set state in the initial position.
        '''
        self.current = (0, 1)  # Row 0, Column 1
        self.previous = None

    def step(self, action: str):
        """
        Performs a movement in the environment and gets the new State and Reward
        ------------
        action : str
          One of the following actions: up, down, left, right
        """
        self.current = self.move(action)

        return self.current, self.get_reward(), self.is_terminal_state()

    def move(self, action: str):
        """
        Calculates the new State given a current State and an Action
        -----------
        action: str
          One of the following actions: up, down, left, right
        """
        self.previous = self.current

        if action not in self.actions:
            raise Exception(f"'{action}' is not a valid action!")

        row = self.current[0]
        column = self.current[1]

        if action == 'up':
            return self.current if self.is_wall(row - 1, column) else (row - 1, column)
        elif action == 'down':
            return self.current if self.is_wall(row + 1, column) else (row + 1, column)
        elif action == 'left':
            return self.current if self.is_wall(row, column - 1) else (row, column - 1)
        elif action == 'right':
            return self.current if self.is_wall(row, column + 1) else (row, column + 1)

    def is_wall(self, row: int, column: int):
        '''
        Returns if the specific row, column is a wall of the Maze
        '''
        return self.maze[row][column] == 'w'

    def get_reward(self):
        """
        Gets Reward of the last movement
        """
        if self.is_terminal_state():
            return 500

            # Returns -2 if the movement was over a Wall, -1 in all other cases
        return -2 if self.previous == self.current else -1

    def is_terminal_state(self):
        '''
        If the State is in the last row, that means that is is in the exit
        '''
        return self.current[0] == self.num_rows - 1

    def set_state(self, row: int, column: int):
        """
        Forces change to a specific state
        """
        if not self.is_wall(row, column):
            self.current = (row, column)
