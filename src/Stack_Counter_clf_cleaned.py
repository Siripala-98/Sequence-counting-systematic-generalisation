"""
Discrete Non-Negative Counter (DNNC)
======================================
A custom PyTorch autograd function implementing a differentiable discrete counter
that tracks non-negative integer counts only.

This is the baseline counter module from El-Naggar (2023), reproduced here for
direct comparison against the DFRC. The DNNC enforces a non-negativity constraint:
when a pop operation is attempted on an empty stack (count == 0), instead of
decrementing the count below zero, a "false pop" is recorded in a separate tracker.

Architecture (2-stack design):
    Stack[0]: Non-negative integer count. Push increments, pop decrements (floor 0).
    Stack[1]: False pop count. Increments when a pop is attempted on an empty stack.

The false pop count acts as a proxy for detecting invalid sequences — a non-zero
false pop count at the end of a sequence indicates unmatched closing brackets.

Backward pass:
    Gradients are computed manually. The push/pop branching logic means the
    gradient through state[0] depends on whether the stack was empty at the
    time of a pop (state[0] == 0 routes gradient to state[1] instead).

References:
    El-Naggar, N. (2023). What Really Counts: Theoretical and Practical Aspects
    of Counting Behaviour in Simple RNNs.
    https://github.com/nadineelnaggar/DNNC
"""

import torch
import torch.nn as nn
from torch.autograd import Function

device = 'cpu'


class StackCounter(Function):
    """
    Custom autograd function implementing the DNNC forward and backward passes.

    Push/pop signals from the preceding linear layer are binarised at a threshold
    of 0.5. The larger signal wins; ties default to NoOp.
    """

    @staticmethod
    def forward(ctx, input, state):
        """
        Args:
            input (Tensor): Shape [2]. input[0] = push signal, input[1] = pop signal.
            state (Tensor): Shape [2]. state[0] = count, state[1] = false pop count.

        Returns:
            output (Tensor): Updated [count, false_pop_count].
        """
        ctx.save_for_backward(input, state)
        output = state.clone()

        threshold = 0.5
        op = 'NoOp'

        # Binarise: whichever signal exceeds threshold and is strictly larger wins
        if input[0] >= threshold and input[0] >= input[1]:
            op = 'Push'
        elif input[1] >= threshold and input[1] > input[0]:
            op = 'Pop'

        if op == 'Push':
            output[0] = state[0] + 1
        elif op == 'Pop':
            if state[0] > 0:
                output[0] = state[0] - 1       # normal decrement
            elif state[0] == 0:
                output[1] = state[1] + 1       # false pop: stack already empty

        ctx.op = op
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Manual backward pass.

        Push always routes gradient to stack[0].
        Pop routes to stack[0] if count > 0, or to stack[1] (false pop) if count == 0.
        """
        input, state = ctx.saved_tensors
        grad_input = grad_output.clone()

        # Input gradients
        grad_push = torch.tensor(1.0) * grad_input[0]
        if state[0] == 0:
            grad_pop = torch.tensor(0.0) * grad_input[0] + torch.tensor(1.0) * grad_input[1]
        else:
            grad_pop = torch.tensor(-1.0) * grad_input[0] + torch.tensor(0.0) * grad_input[1]
        grad_input = torch.tensor([grad_push, grad_pop], requires_grad=True)

        # State gradients
        grad_state = grad_output.clone()

        if ctx.op == 'Push' or ctx.op == 'NoOp':
            grad_state_count = grad_state[0]
            grad_state_falsepop = grad_state[1]
        elif ctx.op == 'Pop':
            if state[0] == 0:
                # Pop on empty stack: gradient flows to false pop tracker
                grad_state_count = torch.tensor(0.0) * grad_state[0]
                grad_state_falsepop = grad_state[1] + torch.tensor(-1.0) * grad_state[0]
            else:
                grad_state_count = grad_state[0]
                grad_state_falsepop = grad_state[1]
        else:
            grad_state_count = grad_state[0]
            grad_state_falsepop = grad_state[1]

        grad_state = torch.tensor(
            [grad_state_count, grad_state_falsepop], requires_grad=True
        )

        return grad_input, grad_state


class StackCounterNN(nn.Module):
    """
    Stateful wrapper around StackCounter (DNNC).

    Maintains count and false-pop-count state between timesteps.
    Resets between sequences.
    """

    def __init__(self):
        super(StackCounterNN, self).__init__()
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.stack = StackCounter.apply

    def reset(self):
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)

    def forward(self, x, y=None):
        """
        Args:
            x (Tensor): Push/pop signals. Shape [2].
            y (Tensor, optional): Initial stack state [count, false_pop_count].

        Returns:
            x (Tensor): Updated stack state [count, false_pop_count].
        """
        if y is None:
            y = torch.tensor(
                [self.stack_depth, self.false_pop_count], requires_grad=True
            )

        x = self.stack(x, y)

        self.reset()
        self.stack_depth += x[0]
        self.false_pop_count += x[1]

        return x

    def editStackState(self, new_stack_depth, new_false_pop_count):
        """
        Manually set the stack state (used in PFC experiments).

        Enforces non-negativity on both values.

        Args:
            new_stack_depth (float): Count to initialise to (clamped to >= 0).
            new_false_pop_count (float): False pop count (clamped to >= 0).
        """
        self.stack_depth = torch.tensor(
            max(new_stack_depth, 0), dtype=torch.float32, requires_grad=False
        )
        self.false_pop_count = torch.tensor(
            max(new_false_pop_count, 0), dtype=torch.float32, requires_grad=False
        )
