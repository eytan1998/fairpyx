"""
An implementation of algorithm in:
2020, Aziz, Simultaneously achieving ex-ante and ex-post fairness
https://arxiv.org/abs/2004.02554

Author: Eitan Ankri, Or Kfir
Date: 2020-04-02
"""
import copy
import logging
import math
from typing import Dict, Any

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from fairpyx import Instance, AllocationBuilder

logger = logging.getLogger(__name__)


def bikroft(matrix: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """
    Birkhoff Algorithm 3:  given  matrix,
    return list of (scalar,matrix) the sclar in (0,1] and matrix is only one 1 in each row and col.
    if you sum the matrix*scalar you get the input matrix!!
    :param matrix: square bistochastic non-negitive Matrix.
    :return P: deterministic allocations represented by permutation matrices with probability for each matrix

==========check that get square matrix==========
    >>> bikroft(np.array([[]]))
    Traceback (most recent call last):
      ...
    ValueError: must be a square matrix
    >>> bikroft(np.array([0]))
    Traceback (most recent call last):
      ...
    ValueError: must be a square matrix
    >>> bikroft(np.array([[0,0],[0,0],[0,0]]))
    Traceback (most recent call last):
      ...
    ValueError: must be a square matrix
    >>> bikroft(np.array([[1]])) == [(1, np.array([[1]]))]
    True

==========check that get non-negitive matrix==========
    >>> bikroft(np.array([[-0.0001]]))
    Traceback (most recent call last):
      ...
    ValueError: must be non-negative
    >>> bikroft(np.array([[-23123]]))
    Traceback (most recent call last):
      ...
    ValueError: must be non-negative

==========check that get bistochastic matrix==========
    >>> bikroft(np.array([[0.1]]))
    Traceback (most recent call last):
      ...
    ValueError: must be a bistochastic matrix
    >>> bikroft(np.array([[0,1],[1,0.0000000000001]]))
    Traceback (most recent call last):
      ...
    ValueError: must be a bistochastic matrix



    """
    # check is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("must be a square matrix")
    if np.min(matrix) < 0:
        raise ValueError("must be non-negative")
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    if not (np.all(row_sums == 1) and np.all(col_sums == 1)):
        raise ValueError("must be a bistochastic matrix")

    P = []
    M0 = np.zeros((matrix.shape[0], matrix.shape[1]))

    while not np.allclose(matrix, M0):
        # Replace zeros with a large value
        modified_matrix = np.where(matrix == 0, np.Inf, matrix)
        # Apply linear_sum_assignment to find the optimal assignment
        row_indices, col_indices = linear_sum_assignment(modified_matrix)
        # Find the edges corresponding to the matched pairs
        edges = [(row, col) for row, col in zip(row_indices, col_indices) if matrix[row, col] != 0]

        scalar = min(matrix[x][y] for x, y in edges)

        P_tmp = np.zeros_like(matrix)
        for x, y in edges:
            P_tmp[x, y] = 1
        P.append((scalar, P_tmp))
        matrix = matrix - P_tmp * scalar
    return P


def who_prefer(instance, item, objects) -> list:
    agent_that_want = []
    for agent in instance.agents:
        ranking = instance.agent_ranking(agent)
        if min([key for key in ranking if key in objects], key=ranking.get) == item:
            agent_that_want.append(agent)
    return agent_that_want


def dict_to_matrix(prob_dict):
    agents = list(prob_dict.keys())
    items = list(next(iter(prob_dict.values())).keys())

    # Create a matrix with the same shape
    matrix = np.zeros((len(agents), len(items)))

    # Populate the matrix
    for i, agent in enumerate(agents):
        for j, item in enumerate(items):
            matrix[i, j] = prob_dict[agent][item]

    return matrix


def PS(instance: Instance) -> dict[Any, dict[Any, int]]:
    """
    Algorithm 2: PS take agents, iteams, and prefrence of agent on iteams.
    and return probobility matrix that say the probobilty for agent i get iteam j.

    :param instance: Represents an instance of the fair course-allocation problem.
    if given prefrence is not stricltly order Instance will break the ties
    :return: p the random assignment

    classic coin toss
    >>> PS(Instance({"avi":{"iteam":1},"beni":{"iteam":1}}))
    {'avi': {'iteam': 0.5}, 'beni': {'iteam': 0.5}}

    only one to allocate to
    >>> num_item = 20
    >>> np.allclose(dict_to_matrix(PS(Instance({"avi":{str(i):i for i in range(num_item)}}))),np.array(1))
    True

    the eaxample from the artical summary and personnaly checked
    >>> np.allclose(dict_to_matrix(PS(Instance(valuations={"avi": {"algo": 3, "inf1": 2, "inf2": 1, "DS": 4},
    ...                               "beni": {"inf1": 3, "inf2": 2, "algo": 4, "DS": 3},
    ...                                "gadi": {"inf1": 1, "inf2": 3, "algo": 2, "DS": 4}})))
    ...                             ,np.array([[0.25  ,     0.5     ,   0.08333333, 0.5       ],
    ...                                         [0.75  ,     0.5     ,   0.08333333, 0.        ],
    ...                                         [0.    ,     0.      ,   0.83333333 ,0.5       ]]))
    True
    >>> np.allclose(dict_to_matrix(PS(Instance(valuations={"avi": {"a": 4, "b": 3, "c": 2, "d": 1},
    ...                                     "beni": {"a": 4, "b": 2, "c": 3, "d": 1}})))
    ...                             ,np.array([[0.5, 1, 0, 0.5],
    ...                                          [0.5, 0, 1, 0.5]]))
    True
    >>> np.allclose(dict_to_matrix(PS(Instance(valuations={"avi": {"a":3, "b": 2, "c": 1},
    ...                                     "beni": {"a":3, "b": 1, "c": 2},
    ...                                      "gadi": {"a": 2, "b": 3, "c": 1}})))
    ...                             ,np.array([[0.5, 0.25, 0.25],
    ...                                        [0.5, 0, 0.5],
    ...                                        [0,0.75,0.25]]))
    True
    >>> d =  Instance({"avi": {"a":3, "b": 2, "c": 1}, "beni": {"a":3, "b": 1, "c": 2},"gadi": {"a": 2, "b": 3, "c": 1}})
    >>> is_envy_free(PS(d),d)
    True
    """

    stage = 0
    O = [instance.items]
    T = [{key: 0.0 for key in instance.items}]
    Tmin = [0.0]
    P = {agent: {item: 0 for item in instance.items} for agent in instance.agents}

    # run until no objects remain
    while any(O[stage]):
        # read to be eaten object - most desirable
        MaxN = set()
        for agent in instance.agents:
            ranking = instance.agent_ranking(agent)
            MaxN.add(min([key for key in ranking if key in O[stage]], key=ranking.get))

        # how much each object is been eaten by one agent. the min is the one that most people eat so end first
        # or how much time take to eat the object
        T.append({key: 0 for key in instance.items})
        for item in MaxN:
            T[stage + 1][item] = ((1 - sum(P[agent][item] for agent in instance.agents))
                                  / len(who_prefer(instance, item, O[stage])))
        Tmin.append(min(value for key, value in T[stage + 1].items() if key in MaxN))

        # Update P of the objects been eaten
        for agent in instance.agents:
            for item in instance.items:
                if agent in who_prefer(instance, item, O[stage]):
                    P[agent][item] = P[agent][item] + Tmin[stage + 1]
        # Remove the eaten objects
        O.append([o for o in O[stage] if o not in [x for x in MaxN if T[stage + 1][x] == Tmin[stage + 1]]])

        stage += 1
    return P


def cut_capacity(G, S):
    capacity = 0
    for u, v in G.edges():
        if u in S and v not in S:
            capacity += G[u][v]['capacity']
    return capacity


def EPS(instance: Instance):
    """
    Algorithm 2(mention): EPS take agents, items, and prefrence of agent on items.
    and return probability matrix that say the probobilty for agent i get item j.

    :param instance: Represents an instance of the fair course-allocation problem.
    not needed strictly order.
    :return: p the random assignment

    ======================like PS======================
    The start is like PS, because if prefrence are stricly order it return the same
    classic coin toss
    >>> np.allclose(dict_to_matrix(EPS(Instance({"avi":{"iteam":1},"beni":{"iteam":1}}))), np.array([0.5,0.5]))
    True

    only one to allocate to
    >>> num_item = 20
    >>> np.allclose(dict_to_matrix(EPS(Instance({"avi":{str(i):i for i in range(num_item)}}))), 1)
    True

    the eaxample from the artical summary and personnaly checked
    >>> np.allclose(dict_to_matrix(EPS(Instance(valuations={"avi": {"algo": 3, "inf1": 2, "inf2": 1, "DS": 4},
    ...                               "beni": {"inf1": 3, "inf2": 2, "algo": 4, "DS": 3},
    ...                                "gadi": {"inf1": 1, "inf2": 3, "algo": 2, "DS": 4}})))
    ...                             ,np.array([[0.25  ,     0.5     ,   0.08333333, 0.5       ],
    ...                                         [0.75  ,     0.5     ,   0.08333333, 0.        ],
    ...                                         [0.    ,     0.      ,   0.83333333 ,0.5       ]]))
    True
    >>> np.allclose(dict_to_matrix(EPS(Instance(valuations={"avi": {"a": 4, "b": 3, "c": 2, "d": 1},
    ...                                     "beni": {"a": 4, "b": 2, "c": 3, "d": 1}})))
    ...                             ,np.array([[0.5, 1, 0, 0.5],
    ...                                          [0.5, 0, 1, 0.5]]))
    True
    >>> np.allclose(dict_to_matrix(EPS(Instance(valuations={"avi": {"a":3, "b": 2, "c": 1},
    ...                                     "beni": {"a":3, "b": 1, "c": 2},
    ...                                      "gadi": {"a": 2, "b": 3, "c": 1}})))
    ...                             ,np.array([[0.5, 0.25, 0.25],
    ...                                        [0.5, 0, 0.5],
    ...                                        [0,0.75,0.25]]))
    True


    ======================for EPS only======================

    """
    # initialize
    round_of_algo = 0
    n = instance.num_of_agents
    m = instance.num_of_items
    P = {agent: {item: 0 for item in instance.items} for agent in instance.agents}
    A = list(instance.items).copy()
    C = [{agent: 0 for agent in instance.agents}]
    # override so wont break ties
    H = {agent:
             {item for item in instance.items if instance.agent_ranking(agent)[item]
              == min((value for key, value in instance.agent_ranking(agent).items() if key in A)) if item in A}
         for agent in instance.agents}

    # setup graph
    G = nx.DiGraph()
    S = 0
    T = m + n + 1
    G.add_node(S)
    G.add_node(T)
    G.add_nodes_from(instance.agents)
    G.add_nodes_from(instance.items)

    G.add_edges_from([(u, T, {'capacity': 1}) for u in A])
    G.add_edges_from([(u, v, {'capacity': np.Infinity}) for u in instance.agents for v in H[u]])

    L_original = 1e-10
    G.add_edges_from([(S, u, {'capacity': C[round_of_algo][u] + L_original}) for u in instance.agents])

    while A:
        L = L_original
        for u in G.successors(S):
            G[S][u]['capacity'] = C[round_of_algo][u] + L

        low = L
        high = instance.num_of_items
        while True:
            mid = (low + high) / 2
            G_TMP = G
            for u in G_TMP.successors(S):
                G_TMP[S][u]['capacity'] = C[round_of_algo][u] + mid
            value, cut = nx.minimum_cut(G_TMP, S, T)
            if cut[0] == {0}:
                low = mid
            else:
                high = mid
                if low > high - 1e-10:
                    low = mid
                    break

        L = low
        for u in G.successors(S):
            G[S][u]['capacity'] = C[round_of_algo][u] + L
        flow, flow_dict = nx.maximum_flow(G, S, T)
        for agent in instance.agents:
            if agent not in cut[0]:
                continue
            for item in flow_dict[agent]:
                if item not in cut[0]:
                    continue
                P[agent][item] = flow_dict[agent][item]

        # Remove all edges that start with nodes in agents
        for agent in instance.agents:
            edges_to_remove = [(agent, v) for u, v in G.edges if u == agent]
            G.remove_edges_from(edges_to_remove)

        G.remove_nodes_from([item for item in A if item in cut[0]])
        A = [x for x in A if x not in cut[0]]

        H = {agent: {item for item in instance.items if instance.agent_ranking(agent)[item]
                     == min((value for key, value in instance.agent_ranking(agent).items() if key in A), default=None)
                     if item in A}
             for agent in instance.agents}

        G.add_edges_from([(u, v, {'capacity': np.Infinity}) for u in instance.agents for v in H[u]])

        C.append({agent: 0 if agent in cut[0] else C[round_of_algo][agent] + L for agent in instance.agents})

        G.remove_nodes_from([agent for agent in instance.agents if H[agent] == set()])

        round_of_algo += 1

    return P


def PS_Lottery(instance: Instance, use_EPS=False):
    """
    Algorithm 1 PS(E)-Lottery that utilize the PS algorithm and Bikroft algorithm to make Simultaneously Achieving Ex-ante and
    Ex-post Fairness.
    :param agents: get number of agents
    :param objects: get number of object
    :param preferences: for each agent need order list corresponding to the most preferred objects (index) to the least,
    it can have set of index if he preferred them the same
    :param use_EPS: difficult false, if true run EPS, if false handel ties with lexicographically order.
    :return: list of deterministic allocation matrix and scalar that represent the chance for this allocation

    # the coin, only one get
    >>> PS_Lottery(Instance({"avi":{"iteam":1},"beni":{"iteam":1}}))
    [(0.5, array([[1.],
           [0.]])), (0.5, array([[0.],
           [1.]]))]
    """
    # assignment
    n = instance.num_of_agents
    m = instance.num_of_items
    c = math.ceil(m / n)
    # 1 make dummy object
    dummy = {"D" + str(x) for x in range(n + 1, n + n * c - m + 1)}
    # 2 add the dummy to the real
    # objects_dummy = instance.items + dummy
    a = Instance({agent: instance.agent_ranking(agent, False) for agent in instance.agents})
    b = {agent: a.agent_ranking(agent, False) for agent in a.agents}
    b = {agent: {d: np.NINF for d in dummy} | a.agent_ranking(agent, False) for agent in instance.agents}
    # 4 new preference
    newInstance = Instance(b)
    # preferences_dummy = [pref + dummy for pref in preferences]
    # 5 run (E)PS then split to presenters
    if use_EPS:
        P = EPS(newInstance) if use_EPS else PS(newInstance)
    else:
        P = PS(newInstance)
    # split to presenters
    extP = [{agent: {item: 0 for item in instance.items | dummy} for agent in instance.agents} for _ in range(c)]
    # for agent in
    for agent in instance.agents:
        ate = 0
        index = 0
        for item in sorted(b[agent].keys(), key=lambda k: b[agent][k], reverse=True):
            # for o in pref:
            if ate + P[agent][item] <= 1:
                extP[index][agent][item] = P[agent][item]
                ate += P[agent][item]
            else:
                extP[index][agent][item] = 1 - ate
                index = index + n
                extP[index][agent][item] = P[agent][item] - (1 - ate)
                ate = P[agent][item] - (1 - ate)
    # after got the matrix from PS, run bikroft
    # and change it to the original agent and object
    combined_mat = dict_to_matrix(extP[0])
    for i in range(1, c):
        combined_mat = np.vstack((combined_mat, dict_to_matrix(extP[i])))

    result = []
    for item in bikroft(combined_mat):
        remove_dummy = item[1][:, :-len(dummy)]
        stack_agents = np.array(np.sum([remove_dummy[row::n] for row in range(n)], axis=1))
        result.append((item[0], stack_agents))
    return result

    def calculate_expected_utility(prob_matrix, preferences, agent):
        """
        Calculate the expected utility of an agent given the probability matrix and preferences.

        :param prob_matrix: Probability matrix as a dictionary of dictionaries.
                            Example: {agent: {item: probability}}
        :param preferences: Preferences as a dictionary of dictionaries.
                            Example: {agent: {item: preference_value}}
        :param agent: The agent for whom to calculate the expected utility.
        :return: Expected utility value.
        """
        expected_utility = 0
        for item, prob in prob_matrix[agent].items():
            utility = preferences[agent][item]
            expected_utility += utility * prob
        return expected_utility

    def is_envy_free(prob_matrix, preferences):
        """
        Check if the given allocation (probability matrix) is envy-free.

        :param prob_matrix: Probability matrix as a dictionary of dictionaries.
                            Example: {agent: {item: probability}}
        :param preferences: Preferences as a dictionary of dictionaries.
                            Example: {agent: {item: preference_value}}
        :return: True if the allocation is envy-free, False otherwise.
        """
        agents = prob_matrix.keys()

        for i in agents:
            my_expected_utility = calculate_expected_utility(prob_matrix, preferences, i)

            for j in agents:
                if i != j:
                    other_expected_utility = calculate_expected_utility(prob_matrix, preferences, j)
                    if other_expected_utility > my_expected_utility:
                        return False

        return True


if __name__ == '__main__':
    import doctest

    print("\n", doctest.testmod(), "\n")

    # instance = Instance(valuations={"avi": {"algo": 3, "inf1": 2, "inf2": 1, "DS": 4},
    #                                 "beni": {"inf1": 3, "inf2": 2, "algo": 4, "DS": 3},
    #                                 "gadi": {"inf1": 1, "inf2": 3, "algo": 2, "DS": 4}})
    # print(instance.agent_ranking("avi"))
    # print(PS(instance))
