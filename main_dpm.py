import numpy as np
from util.multinoulli_class import Multinoulli
from util.dpm_class import DPM
from scipy.stats import beta

np.random.seed(123)


def generate_dataset(num_points, true_k, dim=2, verbose=False):
    """
    Generates num_points datapoints with true_k clusters
    :param num_points: Number of data points
    :param true_k: Number of clusters in the generated data
    :param dim: dimensionality of the data space
    :param verbose: print stuff
    :return: dataset in [num_points,dim]
    """
    # Construct a dataset
    true_z = [i % true_k for i in range(num_points)]

    # I like to sample from the peaked beta distro to get more interpretable results. But feel free to sample
    # from the random uniform
    if False:
        probs_per_cluster = np.random.rand(dim, true_k)
    else:
        probs_per_cluster = beta.rvs(0.5, 0.5, size=(true_k, dim))
    if verbose:
        print('Probabilities for the generated data')
        for i, trueprobs in enumerate(probs_per_cluster):
            print('Cluster %3i with %s' % (i, ' - '.join(['%5.1f' % prob for prob in trueprobs])))
        print('-'*30)

    probs = probs_per_cluster[true_z, :]

    # Sample from the probabilities
    data = (probs > np.random.rand(num_points, dim)).astype(np.float32)
    return data


if __name__ == "__main__":
    """Make some data"""
    dim = 5  # dimensionality of the data
    true_k = 3  # true number of clusters for data generation
    N_start = 500  # number of data points

    data = generate_dataset(N_start, true_k, dim, verbose=True)

    """Set up the priors on the data"""
    # The multinoulli is a conjugate pair with the multibeta distribution
    q0 = Multinoulli(dim=dim, beta=1, gamma=1)

    """Set up a Mixture model for the Dirichlet process"""
    alpha = 1  # concentration parameter for the DP process. See Murphy eq25.17 pg 884
    initial_guess_K = 2  # initial guesses to make for number of clusters
    # random initial assignments
    z = np.random.randint(0, initial_guess_K, (N_start,))
    # Set up the Dirichlet process mixture model
    dpm = DPM(initial_guess_K, alpha, q0, data, z)

    """Start the trial"""
    numstep = 100

    burn_in = 10
    sample_every = 3
    num_clusters = []
    for step in range(1, numstep):
        dpm.step()

        if False:
            print('At step %4i/%4i KK is %i' % (step, numstep, len(dpm.N_k)))

        if step > burn_in:
            if step % sample_every == 0:
                num_clusters.append(len(dpm.N_k))  # Keep track of the number of clusters

    print('Average number of clusters %5.1f \n' % (np.mean(num_clusters)))
    print(dpm.print_probs())
