"""
Discrete Full Range Counter (DFRC)
===================================
A custom PyTorch autograd function implementing a differentiable discrete counter
that supports the full integer range (positive and negative).

This module extends El-Naggar's (2023) Discrete Non-Negative Counter (DNNC), which
could only track non-negative counts. The DFRC generalises this to track signed counts,
which is necessary for recognising Dyck-1 sequences where early closing brackets must
be detected (i.e. the count going negative signals an invalid sequence).

Architecture (2-stack design):
    Stack[0]: Tracks the running integer count (+1 on push, -1 on pop)
    Stack[1]: Tracks a smooth category signal via tanh(Stack[0])
              This provides a differentiable [-1, 1] signal instead of a hard
              positive/negative switch, allowing gradients to flow through
              sign changes during backpropagation.

Backward pass:
    Uses the tanh derivative (1 - tanh²) to propagate gradients from Stack[1]
    back into Stack[0], enabling the model to learn from the sign of the count.

References:
    El-Naggar, N. (2023). What Really Counts: Theoretical and Practical Aspects
    of Counting Behaviour in Simple RNNs.
    https://github.com/nadineelnaggar/DNNC
"""

import torch
import torch.nn as nn
from torch.autograd import Function

device = 'cpu'


class StackCounter_DFRC(Function):
    """
    Custom autograd function implementing the DFRC forward and backward passes.

    The forward pass binarises the RNN's push/pop outputs and updates the
    two-stack state. The backward pass computes gradients manually, using
    the tanh derivative to connect stack[1] gradients back to stack[0].
    """

    @staticmethod
    def forward(ctx, input, state):
        """
        Args:
            input (Tensor): Shape [2]. input[0] = push signal, input[1] = pop signal.
            state (Tensor): Shape [2]. state[0] = current count, state[1] = tanh(count).

        Returns:
            output (Tensor): Updated [count, tanh(count)].
        """
        ctx.save_for_backward(input, state)
        output = state.clone()

        # Binarise push/pop: whichever signal exceeds 0.5 and is strictly larger wins
        threshold = 0.5
        op = 'NoOp'

        if input[0] >= threshold and input[0] >= input[1]:
            op = 'Push'
        elif input[1] >= threshold and input[1] > input[0]:
            op = 'Pop'

        # Update stack[0]: integer count over full range
        if op == 'Push':
            output[0] = state[0] + 1
        elif op == 'Pop':
            output[0] = state[0] - 1

        # Update stack[1]: smooth category signal via tanh
        output[1] = torch.tanh(output[0])

        ctx.op = op
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Manual backward pass.

        Gradients for the push/pop inputs are +1/-1 scaled by the upstream gradient
        on stack[0]. Gradients for the state use the tanh derivative to propagate
        from stack[1] back into stack[0].
        """
        input, state = ctx.saved_tensors

        grad_output_stack_depth = grad_output[0].clone().detach()
        grad_output_stack2 = grad_output[1].clone().detach()

        # Input gradients: push increases count (+1), pop decreases it (-1)
        grad_push = torch.tensor(1.0) * grad_output_stack_depth
        grad_pop = torch.tensor(-1.0) * grad_output_stack_depth
        grad_input = torch.tensor([grad_push, grad_pop], requires_grad=True)

        # State gradients: stack[0] receives gradient from both stack[0] (linear)
        # and stack[1] (via tanh derivative: d/dx tanh(x) = 1 - tanh²(x))
        grad_state_stackdepth = (
            grad_output[0].clone()
            + grad_output_stack2 * (1 - torch.tanh(state[0]) ** 2)
        )
        grad_state_stack2 = grad_output[1].clone()
        grad_state = torch.tensor(
            [grad_state_stackdepth, grad_state_stack2], requires_grad=True
        )

        return grad_input, grad_state


class StackCounterDFRC_NN(nn.Module):
    """
    Stateful wrapper around StackCounter_DFRC.

    Maintains the running stack state between timesteps and resets it
    between sequences. Exposes editStackState() for manual initialisation
    (used in fixed-parameter counter experiments).
    """

    def __init__(self):
        super(StackCounterDFRC_NN, self).__init__()
        self.stack_depth = torch.tensor([0, 0], dtype=torch.float32, requires_grad=False)
        self.stack = StackCounter_DFRC.apply

    def reset(self):
        self.stack_depth = torch.tensor([0, 0], dtype=torch.float32, requires_grad=False)

    def forward(self, x, y=None):
        """
        Args:
            x (Tensor): Push/pop signals from the preceding linear layer. Shape [2].
            y (Tensor, optional): Initial stack state. Defaults to current stack_depth.

        Returns:
            x (Tensor): Updated stack state [count, tanh(count)].
        """
        if y is None:
            y = self.stack_depth.clone().detach().requires_grad_(True)

        x = self.stack(x, y)

        self.reset()
        self.stack_depth[0] += x[0]   # running count
        self.stack_depth[1] = x[1]    # tanh(count)

        return x

    def editStackState(self, new_stack_depth, new_stack2=None):
        """
        Manually set the stack state (used in PFC experiments).

        Args:
            new_stack_depth (float): The integer count to initialise to.
            new_stack2 (float, optional): Ignored; always computed as tanh(new_stack_depth).
        """
        new_stack2 = torch.tanh(torch.tensor(new_stack_depth))
        self.stack_depth = torch.tensor(
            [new_stack_depth, new_stack2.item()], dtype=torch.float32, requires_grad=False
        )
