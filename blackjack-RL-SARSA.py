"""
    Blackjack Expected SARSA Reinforcement Learning
    ===============================================
    Expected SARSA Reinforcement Learning Implementation for Blackjack in Python 
    a = (0, 1]
    e > 0
    Expected SARSA:
        https://en.wikipedia.org/wiki/Reinforcement_learning
        https://en.wikipedia.org/wiki/Q-learning
        https://paperswithcode.com/method/expected-sarsa
        https://tinyurl.com/2p85rauh
        ...
        
"""

import random

ALPHA = 0.3
GAMMA = 0.95
EPSILON = 0.1
EPISODES = 1000

Q = { 'A': [i for i in range(1, 11)] + [10 for i in range(3)],
      'S': [],
      'Q': {},
      'Q done': False,
      'Q reward': 0
    }

def e_greedy(S_):
    Q_values = [Q['Q'].get((tuple(S_), A_), 0.0) for A_ in Q['A']]
    Q_max = max(Q_values)
    Q_dup = Q_values.count(Q_max)
    if Q_dup > 1:
        A_indexes = [i for i in range(len(Q['A'])) if Q_values[i] == Q_max]
        A_index = random.choice(A_indexes)
    else:
        A_index = Q_values.index(Q_max)
    return Q['A'][A_index]

def a_prob(S_):
    S_next_prob = [EPSILON/len(Q['A'])] * len(Q['A'])
    A_max = e_greedy(S_)
    S_next_prob[A_max] += 1 - EPSILON
    return S_next_prob
    
def update(S_, A_, R, S_next):
    S_next_prob = a_prob(S_next)
    Q_next = [Q['Q'].get((tuple(S_), A_), 0.0) for A_ in Q['A']]
    E_next = sum([a + b for a, b in zip(S_next_prob, Q_next)])

    Q_ = Q['Q'].get((tuple(S_), A_), None)
    if Q_ is None:
         Q['Q'][(tuple(S_), A_)] = R
    else:
         Q['Q'][(tuple(S_), A_)] = Q_ + ALPHA * (R + GAMMA * E_next - Q_)

def step(A_):
    if sum(Q['S']) + A_ < 21:
        Q['S'].append(A_)
        Q['Q done'] = False
    else:
        if sum(Q['S']) + A_ == 21:
            Q['Q reward'] += 1
            
        elif sum(Q['S']) + A_ > 21:
            Q['Q reward'] -= 1
        Q['S'] = []
        Q['Q done'] = True
    return Q['S'], Q['Q reward'], Q['Q done']

"""
    Q(s, a) for s is a member of S+
                a is a member of A(s)
    except Q(terminal;) = 0
"""

for i in range(EPISODES):
    Q['S'] = []
    Q['Q done'] = False
    Q['Q reward'] = 0
    ep_reward = 0
    ep_actions = []
    while Q['Q done'] == False:
        if random.random() < EPSILON:
            A_ = random.choice(Q['A'])
        else:
            A_ = e_greedy(Q['S'])
        S_next, R, done = step(A_)
        update(Q['S'], A_, R, S_next)
        ep_reward += R
        Q['S'] = S_next
        ep_actions.append(A_)

    if EPSILON > 0.1:
        EPSILON = EPSILON * 0.9

 
