import numpy as np
from scipy.stats import beta


class Multinoulli:
    def __init__(self, dim, beta, gamma):
        """
        Initialize a Multinoulli distro. As a prior, we use the Multibeta distribution, being
        its conjugate prior.

        For the Beta distro, the sufficients statistics are the counts of 1's and 0's. In this implementation
        we track the 1's in self.counts and we track the total number of data in self.num. From that, we can
        infer the counts of the 0's.
        :param dim:
        :param beta:
        :param gamma:
        """
        self.dim = dim
        self.num = 0  # number of points currently assigned to the cluster
        self.counts = np.zeros(shape=(dim,))
        self.beta = beta
        self.gamma = gamma

    def logpred(self, xx):
        """
        Implements equation 25.36 in Murphy (pag 888)
        It calculates p(x_i | x_{-i}, z_{-i}, z_i = k, \lambda)
        In words, we calculate the probability of x_i given that
        - x_{-i}: given all of the points that are not x_i
        - z_{-i}: given all of the assignments of points not i
        - z_i = k: given that x_i chooses this cluster as its cluster
        - \lambda: the parameters of the prior (gamma and beta)

        We calculate in log domain by convention. Although this works just as well in normal domain.
        :param xx:
        :return:
        """
        result = 0

        for i, xd in enumerate(xx):
            if xd > 0:
                result += np.log(self.beta + self.counts[i])
            else:
                result += np.log(self.gamma + (self.num - self.counts[i]))

        result -= self.dim*np.log(self.beta + self.gamma + self.num)
        return result

    def delitem(self, xx):
        """
        Delete sufficient statistics of xx from the distro
        :param xx:
        :return:
        """
        self.num -= 1
        self.counts -= xx  # assumes xx in {0,1}

    def additem(self, xx):
        """
        Add sufficient statistics of xx to the distro
        :param xx:
        :return:
        """
        self.num += 1
        self.counts += xx

    def get_posterior_multinoulli(self, mode='sample'):
        """
        Gets a sample from the posterior Multinoulli distro
        :param mode: either 'sample' from the posterior or get the 'map'
        :return:
        """
        assert mode in ['map', 'sample'], "expected mode 'map' or 'sample'"
        if mode == 'sample':
            probs = beta.rvs(self.counts + self.beta, self.num - self.counts + self.gamma)
        else:
            probs = (self.beta + self.counts)/(self.beta + self.gamma + self.num)
        return probs
