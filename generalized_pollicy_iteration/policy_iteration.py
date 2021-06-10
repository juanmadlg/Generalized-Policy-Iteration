import numpy as np

from generalized_pollicy_iteration.utils import print_v, print_policy


class PolicyIteration:
    def __init__(self, theta: float, discount: float, environment: object, policy: object):
        """
        Initialization
        ---------------
        theta: float
          Threshold for function value improvement
        environment : object
          The Grid World in our case
        policy : object
          The policy that is going to be improved
        """
        self.environment = environment
        self.policy = policy

        self.V = None

        self.theta = theta
        self.discount = discount

    def get_index(self, row, column):
        return row * self.environment.num_columns + column

    def get_position(self, index):
        x = (index // self.environment.num_columns)
        y = index - (x * self.environment.num_columns)

        return x, y

    def policy_evaluation(self):
        """
        Evaluates the current policy calculating the function value for each state
        """
        self.V = np.zeros((self.environment.num_rows * self.environment.num_columns,))
        self.environment.reset()

        end = False
        while not end:
            delta = 0

            for state in range(self.environment.num_rows * self.environment.num_columns):
                v = self.V[state]

                # Gets the max value got from any of the different accion
                self.V[state] = np.max([self.calc_value(state, action) * self.policy.get_action_probs(state, [
                    self.environment.actions.index(action)]) for action in self.environment.actions])
                # Gets the maximum difference between current and previous values
                delta = max(delta, abs(v - self.V[state]))

            # Only ends if the delta is lower than theta (small value)
            end = delta <= self.theta

        return self.V

    def calc_value(self, state: int, action: str):
        """
        Calculates the function value of the state
        """
        (x, y) = self.get_position(state)

        if self.environment.is_wall(x, y):
            return -2

        self.environment.set_state(x, y)
        if self.environment.is_terminal_state():
            return 0

        new_position, reward, _ = self.environment.step(action)
        new_state = self.get_index(*new_position)

        return reward + self.discount * self.V[new_state]

    def policy_improvement(self):
        """
        Improves a Policy using the current Function Value.
        Returns the new version of the Policy and it has been updated or not.
        """

        # Set max value to terminal states in order to force actions to them.
        self.V[self.get_index(*self.environment.terminal_state)] = np.max(self.V) + 1

        policy_stable = True  # Used to check if the Policy has changed or not in this step

        for state in range(len(self.V)):
            old_actions_probs = np.copy(self.policy.state_actions_probs[state])

            if self.environment.is_wall(*self.get_position(state)):
                self.policy.state_actions_probs[state] = [0, 0, 0, 0]
            else:
                self.policy.state_actions_probs[state] = self.get_greedy_actions(state)

            if not np.array_equal(old_actions_probs, self.policy.state_actions_probs[state]):
                policy_stable = False

        return self.policy, policy_stable

    def get_greedy_actions(self, state):
        """
        Returns action probabilities in order to move to the states that will provide the higher value
        """
        state_action_values = self.get_action_values(state)  # What are the value that we could get from current state

        max_action_value = max(state_action_values)  # What is the higher value
        max_value_indices = [i for i, value in enumerate(state_action_values) if
                             value == max_action_value]  # Gets their indices

        # Prepares action probabilites for the ones with the higher value
        action_probs = np.zeros((4,))
        action_probs[max_value_indices] = 1 / (len(max_value_indices) if type(max_value_indices) is list else 1)

        return action_probs

    def get_action_values(self, state):
        """
        Gets the values for each posible action from the current state
        """
        (x, y) = self.get_position(state)
        num_columns = self.environment.num_columns
        num_rows = self.environment.num_rows

        up = self.V[state - num_columns] if state > num_columns - 1 else self.V[state]
        down = self.V[state + num_columns] if state < num_columns * (num_rows - 1) else self.V[state]
        left = self.V[state - 1] if state % num_columns != 0 else self.V[state]
        right = self.V[state + 1] if (state + 1) % num_columns != 0 else self.V[state]

        return [up, down, left, right]

    def execute(self, verbose=False):
        policy_stable = False

        process = 1
        while not policy_stable:
            self.V = self.policy_evaluation()
            self.policy, policy_stable = self.policy_improvement()

            if verbose:
                print(f'Processed Evaluation + Improvement number {process}', end='\r')

                if policy_stable:
                    print("Policy Stable")

                    print_v(self.V, self.environment.num_columns)
                    print_policy(self.environment.maze, self.policy.state_actions_probs, self.environment.num_rows, self.environment.num_columns)

            process += 1

        return self.policy, self.V
