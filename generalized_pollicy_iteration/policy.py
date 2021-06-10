import numpy as np

class Policy:
    def __init__(self, initial_probs: list, rows: int, columns: int):
        '''
        Initializations
        ----------------
        state_actions_probs: list
          List with the initial probabilites for the 4 available actions
        rows : int
          Number of Maze rows
        columns: int
          Number of Maze columns
        '''
        # Copies sames probabilities in all states
        self.state_actions_probs = np.full((rows*columns, 4), initial_probs)
        self.rows = rows
        self.columns = columns

    def get_action(self, environment: object):
        '''
        Returns action in terms of probabilities of the actions for each state (improved each cycle)
        ------------
        environment : object
          Environment to get available actions
        '''
        current_position_index = environment.current[0]*self.columns + environment.current[1]

        return np.random.choice(environment.actions, p=self.state_actions_probs[current_position_index])

    def get_action_probs(self, state, action):
        return self.state_actions_probs[state, action]
