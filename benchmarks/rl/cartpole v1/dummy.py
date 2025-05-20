import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, variance_scaling, orthogonal

# Dummy data
x = jnp.arange(10000)[:, None]
y = 5 + 2 * x

# Model
class PGActorNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        activation = nn.tanh
        init1 = variance_scaling(jnp.sqrt(2), 'fan_avg', 'truncated_normal')
        init2 = variance_scaling(jnp.sqrt(2), 'fan_avg', 'truncated_normal')
        init3 = variance_scaling(0.01, 'fan_avg', 'truncated_normal')

        x = nn.Dense(128, kernel_init=init1, bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(64, kernel_init=init2, bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(1, kernel_init=init3, bias_init=constant(0.0))(x)
        return x

# Setup
rng = jax.random.PRNGKey(1)
network = PGActorNN()
params = network.init(rng, x)
tx = optax.chain(optax.clip_by_global_norm(1), optax.adam(learning_rate=0.001))
t = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

# Loss function
def loss_fn(params):
    y_hat = t.apply_fn(params, x)
    loss = jnp.mean((y - y_hat) ** 2)
    return loss

gradient_fn = jax.value_and_grad(loss_fn)

# Training loop
# for i in range(100):
#     loss, grads = gradient_fn(t.params)
#     t = t.apply_gradients(grads=grads)
#     if i % 10 == 0:
#         print(f"Step {i}, Loss: {loss}")

def train(runner, i):
    t, x, y = runner
    loss, grads = gradient_fn(t.params)
    t = t.apply_gradients(grads=grads)
    return (t, x, y), i

runner = (t, x, y)
t = jax.lax.scan(train, runner, jnp.arange(100), 100)
