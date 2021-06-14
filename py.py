import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class easy21:
    def _init_(self, max_length=3):
        self.max_length = max_length

    def start(self):
        self.player_sum = np.random.choice(10) + 1
        self.dealer_sum = np.random.choice(10) + 1
        self.state = [self.player_sum, self.dealer_sum]
        self.player_goes_bust = False
        self.terminal = False
        self.turn = 0
        self.reward = 0
        print(self.state)
        return self.state

    def player_move(self, action):

        while self.terminal == False:

            if action == 1:

                self.new_card_value = np.random.choice(10) + 1
                self.new_card_colour = np.random.choice([-1, 1])
                self.player_sum += self.new_card_value * self.new_card_colour
                self.state[0] = self.player_sum
                self.turn += 1

                self.player_goes_bust = self.check_bust(self.player_sum)

                if self.player_goes_bust == 1:
                    self.terminal = True
                    print('haha you lost')
                    self.reward = -1
                    self.state = 'terminal'


            else:

                while self.terminal == False:
                    if 0 < self.dealer_sum < 17:

                        self.new_card_value_dealer = np.random.choice(10) + 1
                        self.new_card_colour_dealer = np.random.choice([-1, 1])
                        self.dealer_sum += self.new_card_value_dealer * self.new_card_colour_dealer
                        self.state[1] = self.dealer_sum
                        self.turn += 1


                    else:
                        self.terminal = True

                        self.dealer_goes_bust = self.check_bust(self.dealer_sum)
                        if self.dealer_goes_bust == 1:
                            print('dealer lost')
                            self.reward = 1
                            self.state = 'terminal'
                        else:
                            if self.dealer_sum > self.player_sum:
                                print('dealer wins')
                                self.reward = -1
                                self.state = 'terminal'

                            elif self.dealer_sum == self.player_sum:
                                print('draw')
                                self.reward = 0
                                self.state = 'terminal'
                            else:
                                print('you win')
                                self.reward = 1
                                self.state = 'terminal'

            return self.reward, self.state

    def check_bust(self, sum):
        return bool(sum > 21 or sum < 1)


def monty_carlo(Q, Return, count_state_action, count_state, n_episodes):
    env = easy21()
    for x in range(n_episodes):
        appeared = np.zeros([21, 10, 2])

        actions = []

        states = []

        rewards = []

        start = env.start()
        states.append(start[:])
        while env.terminal == False:
            greedy_action = Q[start[0] - 1, start[1] - 1, :].argmax()
            count_state[start[0] - 1, start[1] - 1] += 1
            epsilon = count_constant / (count_constant + count_state[start[0] - 1, start[1] - 1])
            action = np.random.choice([greedy_action, 1 - greedy_action], p=[1. - epsilon / 2., epsilon / 2.])
            actions.append(action)
            reward, state = env.player_move(action)
            states.append(state[:])
            rewards.append(reward)
        print(states)
        print(actions)
        print(rewards)
        for x in range(len(states) - 1):
            count_state_action[((states[x])[0]) - 1, ((states[x])[1]) - 1, actions[x]] += 1
            Return[((states[x])[0]) - 1, ((states[x])[1]) - 1, actions[x]] += reward
        for y in range(21):
            for x in range(10):
                for d in range(2):
                    if Return[y, x, d] != 0:
                        Q[y, x, d] = Return[y, x, d] / count_state_action[y, x, d]

    print(Q)  # (np.argwhere(Return!=0))
Q = np.zeros([21,10,2])
Return = np.zeros([21,10,2])
count_state_action = np.zeros([21,10,2])
count_state = np.zeros([21,10])
n_episodes = 100000
count_constant = 100
monty_carlo(Q,Return,count_state_action,count_state,n_episodes)
V = np.transpose(Q.max(axis=2))
print (V)
plt.figure(2)

s1 = np.arange(10)+1
s2 = np.arange(21)+1
ss1, ss2 = np.meshgrid(s1, s2, indexing='ij')

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ss1, ss2, V, cmap=cm.coolwarm)

ax.set_xlabel("dealer's first card")
ax.set_ylabel("player's sum")
ax.set_zlabel("state value")
plt.yticks([1, 5, 10])
plt.yticks([1, 7, 14, 21])
plt.title('Optimal State Value Function in Monte Carlo Control for the States in Easy21' , pad = 20)
fig.colorbar(surf, shrink=0.6)
fig.tight_layout()

plt.show()