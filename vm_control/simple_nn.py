from flax import nnx


class SimpleNN(nnx.Module):

    def __init__(self, n_features: int, n_targets: int, n_hidden: int = 100, *, rngs: nnx.Rngs):
        self.n_features = n_features
        self.layer1 = nnx.Linear(n_features, n_hidden, rngs=rngs)
        self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer3 = nnx.Linear(n_hidden, n_targets, kernel_init=nnx.initializers.zeros, rngs=rngs)

    def __call__(self, x):
        x = x.flatten()
        x = nnx.selu(self.layer1(x))
        x = nnx.selu(self.layer2(x))
        x = self.layer3(x)
        return x
