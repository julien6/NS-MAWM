import cma
import numpy as np


class CMAES:
  def __init__(self, num_params, sigma_init=0.1, popsize=16):
    self.num_params = num_params
    self.popsize = max(2, popsize)
    self.es = cma.CMAEvolutionStrategy(num_params * [0.0], sigma_init, {'popsize': self.popsize})
    self.solutions = None

  def ask(self):
    self.solutions = np.asarray(self.es.ask())
    return self.solutions

  def tell(self, reward_table_result):
    self.es.tell(self.solutions, -np.asarray(reward_table_result))

  def current_param(self):
    return np.asarray(self.es.result.xbest)

  def best_param(self):
    return np.asarray(self.es.result.xbest)

  def rms_stdev(self):
    sigma = np.asarray(self.es.result.stds)
    return np.mean(np.sqrt(sigma * sigma))


class OpenES:
  def __init__(self, num_params, sigma_init=0.1, popsize=16, learning_rate=0.01, **kwargs):
    self.num_params = num_params
    self.popsize = popsize
    self.sigma = sigma_init
    self.learning_rate = learning_rate
    self.mu = np.zeros(num_params)
    self.solutions = None
    self.best = np.copy(self.mu)

  def ask(self):
    self.epsilon = np.random.randn(self.popsize, self.num_params)
    self.solutions = self.mu.reshape(1, -1) + self.sigma * self.epsilon
    return self.solutions

  def tell(self, reward_table_result):
    reward = np.asarray(reward_table_result, dtype=np.float32)
    idx = int(np.argmax(reward))
    self.best = np.copy(self.solutions[idx])
    if np.std(reward) > 1e-8:
      reward = (reward - np.mean(reward)) / np.std(reward)
    self.mu += self.learning_rate / (self.popsize * self.sigma) * np.dot(self.epsilon.T, reward)

  def current_param(self):
    return self.mu

  def best_param(self):
    return self.best

  def rms_stdev(self):
    return self.sigma


class PEPG(OpenES):
  pass


class SimpleGA(OpenES):
  pass
