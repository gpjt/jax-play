Just like PyTorch, defines itself in terms of NumPy 

https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html

Obvs doesn't want to be directly compared to NP itself.

JIT - interesting diff

Autodiff natch

Automatic vectorisation -- should be interesting.

"In local or distributed settings" -- auto-multi-GPU?  Multi-node?
"JAX arrays can be *sharded* across multiple devices for parallel computation."
the array has "devices()" rather than "device" and a "sharding" 
property.

Talks about arrays, nothing about tensors so far.  Ah, but NumPy arrays are
multi-dimensional -- they have .shape.  So they can be tensors.  Makes sense,
"array" as a concept includes arbitrary dimensionality.

They're immutable; JAX is functional

Jit is exciting.  Though speedup shown in example isn't great (292-263us), similar
on Laura.

Thing about it not working with different shapes has a confusing explanation
though.  "Must have the same shape" implied to me that eg.

X = jnp.array(np.random.rand(100, 10))
norm_compiled(X)

...then

X = jnp.array(np.random.rand(100, 10, 3))
norm_compiled(X)

...would fail.  

Their example:

def get_negatives(x):
  return x[x < 0]

x = jnp.array(np.random.randn(10))
get_negatives(x)

jit(get_negatives)(x)

Does indeed fail, but when they say 

This is because the function generates an array whose shape is not known at compile time

...it is misleading, maybe?  Need to read more.  Either (a) the function is
recompiled from scratch for every call, so it must be able to work out the shape
of every array at the start, or (b) it's not, and it's capable of holding kind of
static functions to calculate the sizes (eg. inputs are m x n and n x p, we're 
multiplying them, so I can store the rule 'output is m x p')

Hopefully will become clear in JAX 101, which has a more detailed section.


The gradient stuff is interesting.  Trying to backport it to the PyTorch
way of doing things:

* We want to work out dL/dW for a set of inputs, so we do a forward pass
    then a backward pass, which works that out.
* Notably, the inputs don't appear in that derivative!  While I think about
    it as a step on the loss landscape as a whole, it's really only
    *the loss landscape for those inputs*.  The theory is that the shifting
    loss landscape over the dataset we're feeding in will average out to the
    overall loss landscape of what we're trying to model.

So what we're trying to do is something more like:

def loss(weights):
    model.apply_weights(weights)
    output = model(inputs)
    return calculate_loss(outputs, targets)

...where inputs and targets are known and static.

Then we do

gradients = grad(loss)(weights)

That's much closer to the mathematical formalism than the PyTorch way.

Of course, there may be something more procedural like PyTorch that I'll
learn later.




