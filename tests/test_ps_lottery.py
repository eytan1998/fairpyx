"""
Test the utilitarian-matching algorithm.

Programmer: Erel Segal-Halevi
Since:  2023-07
"""
import logging
import pytest
import fairpyx
import numpy as np

NUM_OF_RANDOM_INSTANCES = 10
NUM_OF_RANDOM_RND_MATRIX = 4

logger = logging.getLogger(__name__)


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


def make_bistochastic_matrix(size):
    matrix = np.random.random((size, size))
    rsum = None
    csum = None
    iteration = 0
    while (np.any(rsum != 1)) | (np.any(csum != 1)):
        matrix /= matrix.sum(0)
        matrix = matrix / matrix.sum(1)[:, np.newaxis]
        rsum = matrix.sum(1)
        csum = matrix.sum(0)
        iteration += 1
        if iteration > 100000:
            matrix = np.random.random((size, size))
            rsum = None
            csum = None
            iteration = 0
    return matrix


def is_envy_free(prob_matrix, instance: fairpyx.Instance) -> bool:
    """
    Check if the given allocation (probability matrix) is envy-free.

    :param prob_matrix: Probability matrix as a dictionary of dictionaries.
                        Example: {agent: {item: probability}}
    :param instance: instance of allocation problem. to calculate the utilty
    :return: True if the allocation is envy-free, False otherwise.
    """

    for agent in instance.agents:
        my_expected_utility = instance.agent_fractionalbundle_value(agent, prob_matrix[agent])
        for other in instance.agents:
            if other != agent:
                # check if i prefer j probability
                other_expected_utility = instance.agent_fractionalbundle_value(agent, prob_matrix[agent])
                if other_expected_utility > my_expected_utility:
                    logger.warning("%s envy %s", agent, other)
                    logger.warning("%s,%s", other_expected_utility, my_expected_utility)
                    return False

    return True


def test_bikroft_sum_equal_original():
    for i in range(NUM_OF_RANDOM_RND_MATRIX):
        matrix = make_bistochastic_matrix(4)
        assert np.allclose(sum(scalar * mat for scalar, mat in fairpyx.algorithms.bikroft(matrix)), matrix)


def test_bikroft_only_one_zero():
    for i in range(NUM_OF_RANDOM_RND_MATRIX):
        matrix = make_bistochastic_matrix(4)
        assert all([all([all(value == 0) or any(value == 1) for value in mat]) for scalar, mat in
                    fairpyx.algorithms.bikroft(matrix)])


def test_bikroft_scalar_sum_1():
    for i in range(NUM_OF_RANDOM_RND_MATRIX):
        matrix = make_bistochastic_matrix(4)
        assert np.allclose(sum([scalar for scalar, mat in fairpyx.algorithms.bikroft(matrix)]), 1)


def test_bikroft_scalars_between_0_1():
    for i in range(NUM_OF_RANDOM_RND_MATRIX):
        matrix = make_bistochastic_matrix(4)
        assert all([0 < scalar <= 1 for scalar, mat in fairpyx.algorithms.bikroft(matrix)])


def test_envy_free_allocation():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        instance = fairpyx.Instance.random_uniform(
            num_of_agents=10, num_of_items=20, normalized_sum_of_values=1000,
            agent_capacity_bounds=[20, 20],
            item_capacity_bounds=[1, 1],
            item_base_value_bounds=[1, 1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
        )
        assert is_envy_free(fairpyx.algorithms.PS(instance), instance)
        assert is_envy_free(fairpyx.algorithms.EPS(instance), instance)


def test_PS_and_EPS_return_the_same():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        instance = fairpyx.Instance.random_uniform(
            num_of_agents=10, num_of_items=20, normalized_sum_of_values=1000,
            agent_capacity_bounds=[20, 20],
            item_capacity_bounds=[1, 1],
            item_base_value_bounds=[1, 1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
        )
        assert np.allclose(dict_to_matrix(fairpyx.algorithms.PS(instance))
                           , dict_to_matrix(fairpyx.algorithms.EPS(instance)))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
