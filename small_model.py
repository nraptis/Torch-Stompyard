import torch
import torch.nn as nn

device = "mps"

"""
This layer transforms each input vector of length 10 into a learned feature
representation of length 32. The layer maintains both a weight matrix and a bias
term, and PyTorch will automatically track their gradients. This is the first
step in the network's feature-extraction pipeline.
"""
_input_to_hidden = nn.Linear(10, 32)


"""
ReLU introduces the nonlinearity that allows the model to approximate complex,
curved functions rather than remaining a purely linear transform. It leaves
positive activations untouched while zeroing out negative ones, stabilizing
training and preventing vanishing gradients. This layer has no trainable
parameters but is essential for meaningful depth.
"""
_hidden_activation = nn.ReLU()


"""
This layer compresses the 32-dimensional hidden representation down into a
single scalar output. It learns how to weight and combine the extracted
features into a final prediction. In a real application, this could represent a
regression output, a logit score, or a probability precursor.
"""
_hidden_to_output = nn.Linear(32, 1)


"""
Sequential chains the layers in exact order so that calling model(x) performs
the complete forward pass automatically. Moving the result onto the MPS device
ensures all parameters and buffers reside on the Apple GPU for fast training.
Every parameter created above is now part of a single cohesive model object.
"""
model = nn.Sequential(
    _input_to_hidden,
    _hidden_activation,
    _hidden_to_output
).to(device)


"""
Here we create a synthetic batch of 16 samples, each with 10 features, purely to
exercise the forward and backward pipeline. The values are random, which is fine
because we're not trying to model real data yet. The tensor is sent to the same
device as the model to avoid cross-device inconsistencies.
"""
x = torch.randn(16, 10).to(device)


"""
These are the random target values corresponding to each of the 16 samples.
Even though the data is artificial, it lets us test that the loss, backward
pass, and optimizer all behave correctly. Like `x`, this tensor is moved to the
MPS device to keep operations consistent.
"""
y = torch.randn(16, 1).to(device)


"""
The optimizer holds references to every parameter in the model so it can update
them during training. Stochastic Gradient Descent adjusts parameters in the
direction opposite their gradients, scaled by the learning rate. A rate of 0.01
is a conservative default that keeps updates stable.
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


"""
Gradients accumulate across training steps, so we must clear them before
computing new ones. Without doing this, gradients from earlier passes would
stack, corrupting the learning dynamics. This line resets the `.grad` fields
for every parameter in the model.
"""
optimizer.zero_grad()


"""
Calling the model with input data automatically executes the entire forward
pipeline: linear → ReLU → linear. PyTorch builds a computation graph during this
operation so that backward() later can determine how each parameter influenced
the final output. The result `pred` is a batch of 16 predictions.
"""
pred = model(x)


"""
Here we compute mean squared error explicitly: (pred - y)^2 averaged across the
batch. Reducing the tensor to a single scalar is necessary because backward()
only operates cleanly on scalars. This value reflects how far off the model's
predictions are from the target outputs.
"""
loss = ((pred - y) ** 2).mean()


"""
This runs automatic differentiation, computing gradients for every parameter
that contributed to the forward pass. PyTorch walks the computation graph from
the loss back through each operation, storing gradient values in each
parameter's `.grad` field. These gradients are what the optimizer will use to
improve the model.
"""
loss.backward()


"""
The optimizer applies the gradient updates computed during backward(). Each
parameter is adjusted by subtracting learning_rate * gradient. After this step,
the model is slightly better at predicting y from x, completing one iteration
of training.
"""
optimizer.step()


print("Loss:", loss.item())
print("Completed forward + backward on:", next(model.parameters()).device)
