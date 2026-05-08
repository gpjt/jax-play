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

More from later:

JAX code used within transforms like jax.jit, jax.vmap, jax.grad, etc.
requires all output arrays and intermediate arrays to have static shape:
that is, the shape cannot depend on values within other arrays.

That's clearer.


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

The extensions to grad -- Jacobian matrices[^1], and so on -- are a bit of a
side quest for now.  I was initially concerned that I might need to learn
it in order to work with an LLM -- after all, there's lots of non-scalar
stuff going on in there.  But crucially, the thing we want to actually
differentiate against, the loss, is scalar.  And it doesn't matter what
goes on in the calculation of that loss -- because the end result is scalar, you can
just use grad on it.  Sanity check:

print(grad(jnp.exp)(x_small))
TypeError: Gradient only defined for scalar-output functions. Output had shape: (3,).

...but...

def foo(X):
    return jnp.sum(jnp.exp(X))

grad(foo)(x_small)

Array([1.       , 2.7182817, 7.389056 ], dtype=float32)


Auto-vectorisation

vmap clearly just maps over the vertical (first) axis of the parameter, which
is normally our batch dimension.  I think they do muddy the waters a bit by
throwing in @jit for the "look, this is faster" examples.  Adding @jit
to their "naively batched" makes it faster than the non-jitted manually batched one!
But keeping jittedness constant, the manually batched is faster than the naively
batched one.

Basically, the ordering is correct but the (mis)use of jit exaggerates the
effect.


Pseudorandom numbers

Appears to be to avoid race conditions inherent in global state, plus
multi-device stuff.  Interesting.

Not just a simple relacement for the internal state that changes itself,
though --

key = random.key(43)
print(random.normal(key))
print(random.normal(key))

...prints

0.07520543
0.07520543

So if you want random numbers that change, you need to split it.

Needs thought.  More in the 101 course, but I suspect I'll need to use
it to understand it.


Debugging

The issue here appears to be something like, running JAX code (at least, with jit)
just builds
a "trace" (~= compute graph?), and doesn't do the calculations.  So if you
print intermediate results, you get a placeholder, not the real value it has
during execution.  The actual calculations only happen when you realise the
result, hence the block_until_ready calls we've had to use with the
%timeit stuff.  This fits in with the functional nature of things.

Relevantly:

JAX is a language for expressing and composing transformations of numerical programs. JAX is also able to compile numerical programs for CPU or accelerators (GPU/TPU). JAX works great for many numerical and scientific programs, but only if they are written with certain constraints that we describe below.

So perhaps it's best to think of JAX as being more of a new language that
happens to look like Python.  Execution of the code is just to generate the
trace, which is then compiled to appropriate operations, including CUDA or
whatever TPUs use.  This is executed at certain points -- block_until_ready,
or presumably other things (though not print!)

Obviously this is a driver for it being functionally pure, or at least wanting
functionally pure code.  Side effects will happen when the Python code is executed,
not when a trace that resulted from that code is executed.  Hence this:

def impure_print_side_effect(x):
  print("Executing function")  # This is a side-effect
  return x

# The side-effects appear during the first run
print ("First call: ", jit(impure_print_side_effect)(4.))

# Subsequent runs with parameters of same type and shape may not show the side-effect
# This is because JAX now invokes a cached compilation of the function
print ("Second call: ", jit(impure_print_side_effect)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
print ("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))

...only prints 'Executing function' for the first and the last, because it
is only then that the Python code is re-run.  In the second, it can re-use
the trace for the first one.

jax.lax appears to contain useful replacements for built-in state-maintaining
things like iterators.

But it's not a panacea.  For example:

array = jnp.arange(10)
print(lax.fori_loop(0, 10, lambda i,x: x+array[i], 0)) # expected result 45

...will work.  fori_loop is basically this:

def fori_loop(lower, upper, body_fun, init_val):
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val

...so in this case, init_val is set to 0, then our lambda is called 10 times
with i set to the loop counter and x set to our accumulator, and we get the
sum of 0 to 9.  Most importantly, even if the loop is unwound, this will work.
(Per https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan
"native Python loop constructs in an jit() function are unrolled, leading to large XLA computations")

By contrast, with

iterator = iter(range(10))
print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0))

...then each internal call to next(iterator) should return a different result,

which is not functional.  JAX will assume that it actually is
functional, so when the first call returns zero it will likely cache it and so you
just wind up adding zero to the accumulator each time and wind up with a result
of zero.

The array update stuff really just drops out of needing to be functional,
though I found this interesting:

"However, inside jit-compiled code, if the input value x of x.at[idx].set(y) is not reused, the compiler will optimize the array update to occur in-place."

Now, what "functional" means with complex parameters like objects
is can be complicated.  If foo is an object of type Foo, then can we
cache the result of

bar(foo)

...?  That's important if we're using

@jit
bar(...)

The sharp bit page covers this purely on the basis of methods, but
problem exists for any jit'ed function taking an object.

See the notebook for examples.

1.  By default JAX will assume that it doesn't have any way to tell and
    if it gets something that isn't an array (or a PyTree, see later),
    it barfs.  So we get past that
    by adding in the static_argnums parameter (which can just be a single
    index for a parameter we're saying "it's OK" about, or a list) to the call
    to jit.  (If we're using it as a decorator, then we can use functools.partial
    for that.)  There's also a static_argnames, which looks a little nicer to
    me.

    But that doesn't fix the problem if the objects are mutable.  JAX's caching
    will use __eq__ and __hash__, so calls with the same object will use the internal
    Python object ID for cache lookups.  So if you call a function passing in foo,
    then change its internal state, then call the function again, JAX will just think
    it's a repeated call and return the cached result.

2.  They suggest overriding __eq__ and __hash__ as a "partial solution", but given
    that they hedge that with "so lon as you never mutate your object", which rather
    seems to make it a non-solution.

3.  Their real solution is to make the class a PyTree.  This appears to be a way
    that an object can expose its inner structure in a way that allows JAX to
    understand it and treat it as a collection of perhaps-mutable things.  You
    essentially say "these are the mutable fields" and "these are the immutable
    fields" and then it can cache sanely.

    Importantly, if you do that, then the foo parameter is *no longer static*.
    So you actually should use the simple @jit decorator with it -- no
    static_argnums.


Out of bounds indexing

Their rationale for this (hard to raise from CUDA) surprised me, as
I'm sure I've seen PyTorch receive errors from CUDA.  But perhaps it's
more of a "this is hard" or even "this is hard on TPUs" or something
like that.

Interesting detail with their example of how to use .at[].get to provide NaNs.
Without that, they have this (which clamps the index):

jnp.arange(10)[11]

...but with it, they use this:

jnp.arange(10.0).at[11].get(mode='fill', fill_value=jnp.nan)

Note the 10.0 in the arange.  This makes sense!  If you use 10, you get an array
of ints, and so there's no such thing as NaN, which is an FP concept.  So this:

jnp.arange(10).at[11].get(mode="fill", fill_value=jnp.nan)

...gives you this:

Array(-2147483648, dtype=int32)


rather than

Array(nan, dtype=float32)



Non-array inputs

Interesting.  It looks like the craziness that leads to one node in the JAXPR
is actually the jnp.array:

make_jaxpr(jnp.array)(x)

{ lambda ; a:i32[] b:i32[] c:i32[] d:i32[] e:i32[] f:i32[] g:i32[] h:i32[] i:i32[]
    j:i32[]. let
    k:i32[] = convert_element_type[new_dtype=int32 weak_type=False] a
    l:i32[1] = broadcast_in_dim k
    m:i32[] = convert_element_type[new_dtype=int32 weak_type=False] b
    n:i32[1] = broadcast_in_dim m
    o:i32[] = convert_element_type[new_dtype=int32 weak_type=False] c
    p:i32[1] = broadcast_in_dim o
    q:i32[] = convert_element_type[new_dtype=int32 weak_type=False] d
    r:i32[1] = broadcast_in_dim q
    s:i32[] = convert_element_type[new_dtype=int32 weak_type=False] e
    t:i32[1] = broadcast_in_dim s
    u:i32[] = convert_element_type[new_dtype=int32 weak_type=False] f
    v:i32[1] = broadcast_in_dim u
    w:i32[] = convert_element_type[new_dtype=int32 weak_type=False] g
    x:i32[1] = broadcast_in_dim w
    y:i32[] = convert_element_type[new_dtype=int32 weak_type=False] h
    z:i32[1] = broadcast_in_dim y
    ba:i32[] = convert_element_type[new_dtype=int32 weak_type=False] i
    bb:i32[1] = broadcast_in_dim ba
    bc:i32[] = convert_element_type[new_dtype=int32 weak_type=False] j
    bd:i32[1] = broadcast_in_dim bc
    be:i32[10] = concatenate[dimension=0] l n p r t v x z bb bd
  in (be,) }


Which matches all but the last line of their example.


The second "Dynamic shapes" section is interesting.  On the face of it it's
just "here's the problem" with a better explanation of what thay mean by "static"
when talking about array sizes.

However, the workaround they give is relevant.  It has the feel of something
they've had to add on to deal with the impurity of the work that people actually
have to do.

The disabled-by-default float64 thing seems a tad odd.  I get that it's not
that useful for ML code, but it's certainly strange that it doesn't do it
even if you explicitly ask for it unless you set a flag in your code.


On to JAX 101.

Interesting: the jit's dislike of non-pure-functional stuff is so strong
that it actually drops impure stuff like the appending to a global list
in the example.  There's literall nothing there to match it!

OK, terminology appears to be:

* When you first run the function (which happens in Python), a tracer
    is attached to arguments.  These tracers are used to build up the jaxpr.
    The process of doing that -- that is, the initial execution -- is called
    a trace.  (Previously I'd been imagining "trace" to mean the recorded
    sequence of steps, but that's not the case.)

However, the fact that the Python code is actually executed is something
they regard as an "implementation detail".  That makes sense!  They could
reasonably write an optimising compiler that did all of the same stuff,
and never actually ran the Python.

This becomes particularly interesting with the "if" example they give.
Only the path taken through the branch is recorded by the trace, so the
jaxpr only includes that.  That makes sense!  While an optimising
compiler sees the whole program, JAX only sees the JAX functions you're
calling, so the "if" is invisible to it.

To reiterate, we need to call the function at least once to get the jitted
version -- it is after all just-in-time, not precompilation.

However, it's interesting that *sometimes* it can detect that the control
flow has something it cannot trace -- eg their example of this raising an
error:

def f(x):
    if x > 0:
        return x
    else:
        return 2 * x

try:
    jax.jit(f)(10)
except Exception as ex:
    print(ex)

How does that differ from this:

def log2_if_rank_2(x):
    if x.ndim == 2:
        ln_x = jnp.log(x)
        ln_2 = jnp.log(2.0)
        return ln_x / ln_2
    else:
        return x

print(jax.make_jaxpr(log2_if_rank_2)(jnp.array([1, 2, 3])))

...?  The latter only recorded the branch taken, whereas the `f` function
failed to jit.

Dumb idea, is it the use of == instead of >?  Confirmed not.  Likewise
going through the else branch rather than the if branch is not the problem.

The important thing is that they used .ndim in the working one:

"Traced values within JIT, like x and n here, can only affect control flow via their static attributes: such as shape or dtype, and not via their values."

ndim is clearly one of those static attributes.  And even though you can
affect control flow with them, you might get weird results.

The static stuff makes a comeback.  I'm reading the static_argnums stuff
as meaning "you can use this arg as a key into a cache mapping from its values
to the cached jaxprs that are appropriate when it takes that value".  So if
you call it with 10, then you get a mapping 10->a jaxpr, and then when you call
it with 12, likewise 12->jaxpr.  Which makes their comment that you should only
use it with a limited number of values for that arg make sense.

It also makes sense in terms of all of the equality/hash key stuff above.

This is confirmed in the "JIT and caching" section.


Automatic vectorization

Basically vmap appears to add on an extra dimension (by default the first one)
to parameters (or just some of them) so that you can have your function do a bunch of
operations at once.  Kind of like an auto-broadcast.

However, it's a convenience thing for your own functions.  jnp.matmul handles
broadcast for you.


Automatic differentiation

So, grad(f) works on its own.  You don't need to nail down the actual parameters.
However, it looks like it's lazy; the actual derivative isn't calculated until
you do call it on a parameter (which is necessary to get a real value).  So it's
not some clever symbolic thing.

Their logistic regression example confirms my belief from earlier.  The way you
work out the gradients is:

* Define a loss function that closes over the inputs and the targets, and takes
    the weights as parameters
* Take its gradient and run through the weights
* You can extract the gradients.  If there are multiple weight arguments, you can
    either ask for them all to be returned as a tuple (with argnums) or pass them in as
    a dict and get a dict of grads (plus other clever options using PyTrees)

Their reference to Spivak’s classic Calculus on Manifolds (1965) is telling, as are
the others.  This is a very mathematical model.  Elegant and beautiful.

On the other hand, jax.value_and_grad looks like a neat convenience function
for when you need to log the loss as well as apply the grads.

check_grads looks like a useful internal function, not sure how we might use
it in code ourselves.  Perhaps if we have to do our own "backward pass" stuff?



PyTrees

First thought: if this is easy, maybe I was wrong to semi-dismiss it as
"something clever" in the grad stuff.  My big concern at the moment is:

* I build my LLM.
* I want to be able to differentiate wrt the weights -- they're a parameter
    passed in rather than object state.
* I have to pass in some messy dict and read stuff out as part of the LLM
    execution.

Maybe that wouldn't be too bad, but it would be nice to have a "weights"
object, maybe?  Anyway, a PyTree can represent that.

Interesting point -- in a pytree, the elements of a jnp.array are not leaves.
That makes sense, I think.  You want arrays to be treated as a unit for
eg. backprop purposes.

Aha!  But they give an MLP example.  And this is where it gets really interesting:

@jax.jit
def update(params, x, y):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(
        lambda param, grad: param - LEARNING_RATE * grad, params, grads
    )

Of course!  Because by default the grad function only returns gradients against
the first parameter, and because it matches the shape of that parameter when it's
a dict, then we get something shaped just like the params.

So no need to explicitly close over the inputs and targets, you can provide them
as parameters to the loss function and then grad will just ignore them
when working out gradients.

And the fact that it matches the shape of a dictionary (or rather a PyTree) when
calculating the gradients means that they're appropriately shaped to do the update
with a map!

Moving on.  The "Pytrees and JAX transformations" section is rather poorly explained,
as the code there is not executable.  I *think* I get it but I'm not sure
I can put together code that would test my understanding.
























[^1]: I'm glossing that as "derivatives for functions with a non-scalar result"
    for now -- will learn more later.  Pronounced jac-O-bian, not to be confused
    with eg plays which are jaco-BE-an -- spelling diff as well as pronunication
    and meaning.  Joke in there somewhere.

