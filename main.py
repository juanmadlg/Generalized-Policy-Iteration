import numpy as np
from generalized_pollicy_iteration.mazes import mazes
from generalized_pollicy_iteration.maze_environment import MazeEnvironment
from generalized_pollicy_iteration.policy import Policy
from generalized_pollicy_iteration.policy_iteration import PolicyIteration

if __name__ == "__main__":
    maze = mazes[0] # You could try 0 to 7 different mazes

    env = MazeEnvironment(maze)
    policy = Policy(np.array([0.25, 0.25, 0.25, 0.25]), len(maze), len(maze[0]))

    policy, v = PolicyIteration(theta=0.001, discount=1, environment=env, policy=policy).execute(verbose=True)
