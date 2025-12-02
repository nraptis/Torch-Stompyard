import torch
import torch.nn as nn

device = "mps"

"""
------------------------------------------------------------
PROBLEM STATEMENT (THE PRETEXT)
------------------------------------------------------------
In machine learning, the central task is to learn a function
that maps input data to output targets. Here, we create the
smallest possible neural network that demonstrates this idea:
given some input vectors (10 numbers each), the model tries to
produce output values that match our target values (1 number
each). Even though the data here is randomly generated, the
training process is real: the model must adjust its internal
parameters so that its predictions become closer to the targets.

This example serves as a miniature demonstration of:
- how neural networks are constructed from layers,
- how data flows through them during a forward pass,
- how the loss is computed,
- how backpropagation produces gradients,
- and how an optimizer updates the model parameters.

Every step shown here mirrors the structure of real ML systems,
just scaled down to the simplest possible components.
------------------------------------------------------------
"""


"""
------------------------------------------------------------
LAYER 1: INPUT → HIDDEN TRANSFORMATION
------------------------------------------------------------
This linear layer is the first step in learning a useful
representation from the input data. It takes a vector of 10
numbers and transforms it into a vector of 32 numbers using a
matrix multiplication plus a bias. These 32 outputs are “hidden
features” — patterns the model thinks are useful for prediction.

This layer has trainable weights and biases. During training,
these values will shift to reduce the loss.
------------------------------------------------------------
"""
_input_to_hidden = nn.Linear(10, 32)


"""
------------------------------------------------------------
ACTIVATION FUNCTION: NON-LINEARITY
------------------------------------------------------------
ReLU (Rectified Linear Unit) applies a simple rule:
if a value is > 0, keep it; otherwise, set it to 0.

The purpose is to introduce non-linearity into the model. Without
non-linearity, stacking linear layers would still produce a purely
linear transformation, limiting the model’s ability to learn
complex patterns. ReLU is cheap, effective, and widely used.
------------------------------------------------------------
"""
_hidden_activation = nn.ReLU()


"""
------------------------------------------------------------
LAYER 2: HIDDEN → OUTPUT
------------------------------------------------------------
This linear layer reduces the 32 learned features down to a
single output value. In a real regression problem, this might
represent a predicted price, score, or measurement. This layer
is the part of the network that “summarizes” all extracted
information into one meaningful final prediction.

It also contains learned weights and a learned bias.
------------------------------------------------------------
"""
_hidden_to_output = nn.Linear(32, 1)


"""
------------------------------------------------------------
MODEL CONSTRUCTION
------------------------------------------------------------
Sequential builds a pipeline where data flows through each layer
in order: input_to_hidden → ReLU → hidden_to_output.

Moving the model to the MPS device ensures that all parameters
and calculations occur on the Apple GPU when available. Every
operation involving this model must now place tensors on the
same device.
------------------------------------------------------------
"""
model = nn.Sequential(
    _input_to_hidden,
    _hidden_activation,
    _hidden_to_output
).to(device)


"""
------------------------------------------------------------
SYNTHETIC INPUT DATA
------------------------------------------------------------
Here we generate 16 synthetic samples, each containing 10 input
features. Although random, this data forces the network to perform
actual computation and learn from the backward pass.

Machine learning always relies on a set of input samples; in a
real application, these might be images, measurements, or records.
------------------------------------------------------------
"""
x = torch.randn(16, 10).to(device)


"""
------------------------------------------------------------
SYNTHETIC TARGET OUTPUTS
------------------------------------------------------------
These are the values we want the model to predict. Again, they’re
random—because this example is about demonstrating mechanics, not
solving a real problem.

The network will try to match its predicted values to these
targets by adjusting its parameters.
------------------------------------------------------------
"""
y = torch.randn(16, 1).to(device)


"""
------------------------------------------------------------
OPTIMIZER SETUP
------------------------------------------------------------
SGD (Stochastic Gradient Descent) updates the model’s parameters
using their gradients. The learning rate determines how large
each update step is.

The optimizer’s job is to move the model in a direction that
reduces the loss over time.
------------------------------------------------------------
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


"""
------------------------------------------------------------
TRAINING STEP 1: CLEAR GRADIENTS
------------------------------------------------------------
PyTorch accumulates gradients across training steps. Before
computing new gradients, we must reset the old ones.

This ensures that each backward pass only reflects the current
forward pass.
------------------------------------------------------------
"""
optimizer.zero_grad()


"""
------------------------------------------------------------
TRAINING STEP 2: FORWARD PASS
------------------------------------------------------------
Passing the input data through the model produces a tensor of
predictions. The network processes the data layer by layer,
applying learned transformations and activation functions.

PyTorch builds a computation graph during this operation so
backpropagation knows how to compute gradients later.
------------------------------------------------------------
"""
pred = model(x)


"""
------------------------------------------------------------
TRAINING STEP 3: LOSS CALCULATION
------------------------------------------------------------
Loss measures how far the predictions are from the targets.
Here we compute Mean Squared Error manually:
    (prediction - target)^2 averaged across all samples.

The loss must be a scalar because backward() requires a single
value to differentiate.
------------------------------------------------------------
"""
loss = ((pred - y) ** 2).mean()


"""
------------------------------------------------------------
TRAINING STEP 4: BACKWARD PASS
------------------------------------------------------------
This step performs automatic differentiation. PyTorch walks
backward through the computation graph and calculates gradients
for every trainable parameter. These gradients show the optimizer
how to adjust each weight to reduce the loss.

Without this step, learning cannot occur.
------------------------------------------------------------
"""
loss.backward()


"""
------------------------------------------------------------
TRAINING STEP 5: UPDATE PARAMETERS
------------------------------------------------------------
The optimizer modifies each parameter according to:
    new_value = old_value - learning_rate * gradient

After this update, the model is slightly better at predicting
y from x. This completes one training iteration.
------------------------------------------------------------
"""
optimizer.step()


print("Loss:", loss.item())
print("Completed forward + backward on:", next(model.parameters()).device)
