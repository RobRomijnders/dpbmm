import numpy as np
import copy


class DPM:
    def __init__(self, initial_k, alpha, prior, data, z):
        """
        initialize DP mixture model
        :param initial_k: initial guess for the mixture components
        :param alpha: concentration parameter
        :param prior: function for the Multinoulli prior
        :param data: data as shape [num_samples,dim]
        :param z: initial guess for the cluster assignments
        """
        self.num_clusters = initial_k
        self.num_samples, self.dim = data.shape
        self.alpha = alpha
        self.copy_of_prior = prior  # when initializing a new cluster, we copy this prior to start from
        self.data = data
        self.z = z
        self.N_k = [0]*initial_k  # Number of points in cluster k, like N_k in Murphy eq25.35 pg 888

        self.cluster_distros = []
        # Initialize the priors on the mixture components
        for _ in range(initial_k):
            self.cluster_distros.append(copy.deepcopy(prior))

        self.include_points(data, z)

    def include_points(self, data, z):
        """
        Add all of the data and its assignments, z, to the distributions
        :param data: data with shape [num_samples, dim]
        :param z: the (possbly guessed) cluster assignments, list of integers. length [num_samples,]
        :return:
        """
        for i, x in enumerate(data):
            k = z[i]
            self.cluster_distros[k].additem(x)
            self.N_k[k] += 1
        self.num_samples = sum(self.N_k)

    def step(self):
        """
        Make one step in the collapsed Gibbs sampling
        :return:
        """

        # For one Gibbs sample:
        # 1. Remove one x_i from the model (so remove its sufficient statistics
        # from the cluster it is currently assigned to)
        # 2. Make the distro over clusters for this point (eq 25.33 Muprhy pg 888)
        # 3. Sample from this distro and put x_i in that cluster

        for i, xx in enumerate(self.data):
            # -- 1 --
            k_old = self.z[i]
            self.N_k[k_old] -= 1
            self.cluster_distros[k_old].delitem(xx)
            self.remove_cluster_if_empty(k_old)

            # -- 2 --
            # note that we compute p(z_i=k|z_{-i}, x, alpha, lambda) in one go. (eq25.33 Murphy)
            # We could explicitly separate this computation if we'd like into
            # -- p(z_i=k|z_{-i}, alpha) (eq25.35 Murphy)
            # -- p(x_i | x_{-i}, z_i=k, z_{-i}, lambda) (eq.25.36 Murphy)
            pp = self.N_k.copy()
            pp.append(self.alpha)
            pp = np.log(np.array(pp))
            for k in range(self.num_clusters+1):
                pp[k] += self.logpredictive(k, xx)
            pp = np.exp(pp-np.max(pp))  # Subtract max to avoid numerical errors
            pp /= np.sum(pp)

            # Random sample from the conditional probabilities
            # Corresponds to line10 in algorithm25.7 Murphy pg 889
            # k_new = np.random.choice(self.num_clusters+1, p=pp)
            uu = np.random.rand()
            k_new = int(np.sum(uu > np.cumsum(pp)))

            # -- 3 --
            self.add_cluster_maybe(k_new)

            self.z[i] = k_new
            self.N_k[k_new] += 1
            self.cluster_distros[k_new].additem(xx)

    def add_cluster_maybe(self, k_new):
        """
        Maybe adds a cluster in case we sample a new k.
        In the Gibbs sample, if you draw z_i=k*, then we add a new cluster.
        This is described by Murphy eq25.38 and the subsequent text
        :param k_new:
        :return:
        """
        if k_new == self.num_clusters:
            self.num_clusters += 1
            self.N_k.append(0)
            self.cluster_distros.append(copy.deepcopy(self.copy_of_prior))

    def logpredictive(self, k, xx):
        """
        Calculates the log predictive distro for x_i given all other x_{-i}
        Corresponds to Murphy eq25.36 pg 888

        Note that if k == KK, then the posterior log-predictive corresponds to the prior log predictive.
        (See eq25.37-38 Murphy pg 888)
        :param k: the cluster under consideration. (Corresponds to 'z_i=k' in eq25.36)
        :param xx: the x_i under consideration
        :return:
        """
        if not k == self.num_clusters:
            q = self.cluster_distros[k]
        else:
            q = copy.deepcopy(self.copy_of_prior)
        return q.logpred(xx)

    def remove_cluster_if_empty(self, k):
        """
        If the cluster k is empty, then remove it.

        For example: Some cluster, k, has one data point, x_i, assigned to it. We make a Gibbs sample and remove
        x_i. Then cluster k is empty and we remove it from our state
        :param k:
        :return:
        """
        if self.N_k[k] == 0:
            self.num_clusters -= 1
            self.cluster_distros.pop(k)
            self.N_k.pop(k)
            self.z[np.argwhere(self.z > k)] -= 1

    def print_probs(self):
        """
        Print the clusters and the MAP assignment of its Multinoulli parameters
        :return:
        """
        print('The MAP assignments of the clusters that we sampled')
        for i, k in enumerate(np.argsort(self.N_k)[::-1]):
            q = self.cluster_distros[k]
            map_assignment = q.get_posterior_multinoulli('map')
            print('Cluster %3i with %5i data and MAP %s' %
                  (k, q.num, ' - '.join(['%5.2f' % prob for prob in map_assignment])))
