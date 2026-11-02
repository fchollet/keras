# Keras API design guidelines

These guidelines are meant to help focus design discussions and help us create delightful developer experiences.

These are meant as guidelines, not rules: each decision should be debated in its own unique context.

Some text remixed from external references:

- [User experience design for APIs](https://blog.keras.io/user-experience-design-for-apis.html)
- [Notes to Myself on Software Engineering](https://medium.com/s/story/notes-to-myself-on-software-engineering-c890f16f4e4d)

---

## Design end-to-end workflows, not individual functions and classes.

When developing APIs, start by designing end-to-end workflows, and only sketch out specific function/class signatures at the end.

- The goal is to arrive at workflows that feel like they are purposefully designed and well-optimized, rather than cobbled together to route around the features provided by the API. The workflows should come first, before atomic features. **Features only exist to support a workflow.** No feature should exist to provide a capability "just in case", "because we can".
- **Every design review document should prominently feature a code example of one or two end-to-end workflows showing the canonical use-case for the new API.**
- Every time we discuss choices surrounding a specific API feature, we should start by asking: **in what workflows will this be used?** Then we should make the choice that makes the most sense with respect to these workflows. We should not make API design decisions about features in isolation.
- This implies that we will often ask the question: **do users really need to configure this parameter?**, and in many cases, the answer will be "no", rather than being "yes" by default.

---

## Carefully weigh whether a new feature should be included.

It's okay to say no: just because someone asks for a feature doesn't mean we should do it. Every feature has a cost that goes beyond the initial CL: maintenance cost, documentation cost, and cognitive cost for our users (a sprawling API surface is a major usability issue).

In particular, in the Keras API, every new feature has to be maintained in perpetuity.

As such, our criteria for adding a new feature in the API is the following:

- **It should be broadly useful to our users**, rather than a niche feature that is only relevant to a specific vertical of researchers. Niche features should be maintained independently by those who need them (e.g. by extending the API via subclassing), as third-party add-on packages.
- **It should be widely recognized as a machine learning best practice.** We will not add new layers/etc that were recently published to ArXiv.org, even in case of claims of increased accuracy/etc. We only add new objects that are already commonly used in the machine learning community. Presumably, a new technique that does result in meaningful gains would be broadly adopted after a few months anyway (like ResNet), and that's when we would be adding it to the core API. SIG-addons maintains a repository of significantly more volatile and independently maintained code to which the barriers to entry are lower.
- **It should have an owner committed to maintaining it in the long term.** In particular, the code should be maintainable by multiple people on the team, not just by one technical guru.

In addition, when saying yes to a request for supporting a new use case, remember that **literally adding what the user/team requested is often not the optimal choice**. Users are focused on their own specific use case, and we must counter this with a holistic and principled vision of the whole project (see: designing end-to-end workflows, not atomic functions/classes). Often, the right answer is to extend an existing feature. **Find the natural place to integrate the new feature in existing APIs.**

### Examples:

- We should not have added the self-normalizing activation function to the API. It was added before passing the test of time, and that technique has shown later not to reach broad adoption. **Note that citation count is not a good metric of adoption**; that paper has a high citation count.
- We should not move to core an API that has debuted somewhere on GitHub or TF-Addons but has failed to gain more than a few users after a few months.

---

## Seek to minimize cognitive load for our users.

Always seek to minimize the cognitive load imposed on our users in the course of using our APIs.

At a high level:

- **Automate everything that can be automated.**
- **Minimize the actions & choices required from the user.** Make sure default values for arguments are sensible and reflect best practices (so that users usually wouldn't have to manually configure these). Don't expose options that are not important or do not match real use cases, "just in case".
- **Design simple and consistent workflows that reflect simple and consistent mental models.**

Here are a few practical rules:

- **No API should deal with internal implementation details.** An API is a language for our users to talk about the problem they care about -- and they don't care about our internal hacks. For instance, an option like `use_locking` in an optimizer should be avoided. If an argument requires users to understand the implementation (not just what the code is supposed to implement, like SGD in this case), then the argument should not be included in the public API. **An API is all about the problem it solves, not about how the code works in the background.**
- **Introduce as few new concepts as possible.** It's not just that additional data structures require more effort in order to learn about their methods and properties, it's that they multiply the number of **mental models** that are necessary to grok your API. Ideally, you should only need **a single universal mental model around which everything is organized** (in Keras, that's the `Layer`). Definitely avoid having more than 2 or 3 mental models underlying the workflows you design. Likewise, avoid having concepts that are mostly overlapping but subtly different, since the difference will be difficult to convey clearly and will confuse our users (like, say, `Network` and `Model` -- this is why we don't export `Network` as a public API).
- **Objects that do interchangeable things should have identical or very close APIs.** In particular they should have the same positional arguments. For example, it should be possible to swap one optimizer for another in user code (when leaving all arguments to their default value) without editing the arguments.
- **If you find yourself proposing a signature with more than 6-7 arguments, consider whether all of these arguments are useful.** How many people and use cases would be affected if you removed one argument? How much would they be affected -- would they be able to easily extend the API (e.g. via subclassing) to support their use case without that built-in argument? Could this API be broken up into smaller, modular objects?
- **Best-practices should come baked into your API.** The simplest way to use your API (leaving all arguments to their default value, using the most obvious tool for the task, etc) should be as close as possible to the best way of solving the problem. In particular, all arguments that can be given a default value should be given a default value, and that default should match the most common use case.
- **Plain Python types are preferable to custom types.** Use tuples, strings, ints... A custom type requires more knowledge and effort on the part of the user (e.g. `TensorShape`, which is also breaking established conventions of scientific Python). **When using enums, make sure that their values are strings**, so as to make it possible for users to pass plain strings (example: `data_format="channels_last"`, `padding="valid"`).
- **Explicit, single-level configuration arguments are preferable to nested, hidden configuration arguments.** Avoid something like: `MyLayer(hyperparameter_dict)`, instead use `MyLayer(units, activation=None, ...)`.

In particular, naming is important and difficult:

- **The meaning of an argument should be clear from its name and should not require knowledge that only the implementers have.** In particular, argument names should only involve recognized terms of art ("L1 norm" is a term of art), and should not involve implementation-related vocabulary (e.g. "fused batchnorm").
- **Avoid `OverlyLongAndSpecificNamingPatterns`.** If you find yourself with argument names with involve more than 3 subparts (e.g. "squared_operator_norm"), reconsider. Argument names should be intuitive and easy to remember.
- Avoid overly generic names (`x`, `variable`, `parameter`).
- **Make sure you are consistent in your naming choices.** Naming consistency means both **internal naming consistency** (don't call `dim` what is called `axis` in other places, don't call `ndims` what is called `ndim` elsewhere) and **consistency with established conventions for the problem domain (terms of art)**. Before settling on a name, make sure to look up existing names used by domain experts (or other APIs). In our case, argument names should be consistent with the broader scientific Python conventions, in particular NumPy.

Note that Keras uses the following naming rules:

- We use the convention `num_*` for counters, though omitting an explicit counter is nicer when there is no ambiguity (e.g. `units`, `epochs`, `filters`).
- The rank of a tensor is its `ndim`. A specific dimension index is an `axis`. The number of dimensions in a linear projection (or similar) is `units`.
- By convention Keras layers are named with nouns rather than verbs (e.g. `Normalization` and not `Normalize`, `Convolution` and not `Convolve`).
- Following Python conventions, classes use capitalized parts (e.g. `ClassName`) and functions and methods use snake case (e.g. `function_name`).
- If an argument name has a numerical suffix (e.g. `alpha_1`), we put an underscore before the suffix in snake case. The capitalized equivalent would be e.g. `Alpha1`.
- We used fully spelled-out names, e.g. `attention_scores` and not `attn_scores`. There are a couple standardized exceptions to this rule, in particular `dim` for "dimension" and `num` for "number". These are sufficiently common that they are not ambiguous to a first-time reader.

### Example:

```python
MyConstructor(
   per_variable_sparsity_config=[
      'layer_1/kernel:0.8', 'layer_2/kernel:1.5'])
```

What's wrong with this?

- Overly long argument name
- Too much cognitive load involved in preparing an appropriate argument value
- Preparing an argument value requires internal implementation knowledge
- Reliance on TF variable names (subject to changes at any time, thus breaking this code)
- Nested config adding indirection
- Incorrect typing (float values being passing as strings)

Possible alternative:

```
obj = MyConstructor()
obj.configure_sparsity(some_layer.kernel, value=0.8)
obj.configure_sparsity(some_other_layer.kernel, value=1.5)
```

What's nice about this?

- Object-based variable references.
- Modular, simple action, with a clear name.
- Plain Python types.

---

## Balance expressivity vs. user-friendliness.

### Simple use cases should be simple, advanced use cases should be possible:

**Don't increase the cognitive load of common use cases for the sake of niche use cases**, even minimally.
**Make sure that advanced users have a path to support their use case**, even if this path requires the users to roll out plugins or other API extensions (in particular via subclassing). **It is ok for advanced use cases not to be directly supported in the built-in API options.**

### Keep our APIs modular.

**Complex objects should be achievable by composing simple objects with few arguments, that do one thing reliably.** There is a balance to strike between having complex signatures on fewer objects, and having more objects with simpler signatures. A good API has a reasonable number of objects, with reasonably simple signatures (see also: avoiding signatures with more than 6-7 arguments).

**Things that create state or side-effects should be classes. Functions should be stateless.**
For instance, layers that create weights should not be cast as functions, since it makes the weights (and other elements of state) hard to access, impossible to update, and forces reliance on a global state capturing the side effects of layer-functions.

### APIs should be strictly compartmentalized.

For instance, the optimizer API or the layers API should not contain arguments for configuring distributed training. That should go into the distribution API.

---

## Don't neglect error messages, docstrings, and documentation.

Documentation and error messages are an integral part of the API. Good docs and helpful error messages are key to a delightful user experience.

- **Catch user errors early and anticipate common mistakes.** Do user input validation as soon as possible. Actively keep track of common mistakes that people make (by screening GitHub and StackOverflow), and either solve them by simplifying our API, adding targeted error messages for these mistakes, or having a "solutions to common issues" page in our docs. Consider adding automated fallback behaviors (e.g. casting a wrongly-typed input) instead of raising errors, when applicable. Be nice to our users.
- **Provide detailed feedback messages upon user error.** Error messages should be contextual, informative, and actionable. Every error message that transparently provides the user with the solution to their problem means one less support ticket, multiplied by how many times users run into the same issue. A good error message should answer:
    - What happened, in what context? (Include all relevant hyperparameters, e.g., `strides`, `padding`, `pool_size`, not just the input shape. Always include the *received* values like `original_shape` or `output_mode`).
    - What did the software expect? (Explicitly mention the expected vs. actual path or value for complex models).
    - How can the user fix it?
- **Use plain Python representations in error messages.** Shapes and sizes should be represented as Python tuples (e.g., `(2, 2)`) rather than raw NumPy array representations (e.g., `[2 2]`). Use `.name` attributes for objects instead of raw object strings.
- **A docstring should answer the question: what is this about, and why & how should I use it?** It should assume as little context as possible, and it shouldn't mention specialized terms without first introducing them (for example, "num_blocks: Number of blocks in the kernel" is not a good argument description if this is the first time you mention "blocks" in your docstring).
- **Show, don't tell: your documentation should not talk about how the software works, it should show how to use it.** Show code examples for end-to-end workflows; show code examples for each and every common use case and key feature of your API. **All docstrings should include code examples.**
- **Deliberately design the user onboarding process for your feature.** How are complete newcomers going to find out the best way to solve their use case with your tool? Have an answer ready. Make sure your onboarding material closely maps to what your users care about: don't teach newcomers how your framework is implemented, teach them how they can use it to solve their own problems. After shipping a CL and writing good docstrings, make sure to create a Colab guide / tutorial showcasing the target workflow, and post it on the docs website.
- The feature is not ready until:
    - 1) Users know about it
    - 2) They know how to use it
    - 3) They're actually using it to solve the corresponding problem.

Note that Keras uses the following rules for writing docstrings:

- For class docstrings, document arguments in an `Args:` section in the class docstring, not in `__init__`.
    - When a user creates a class, they are not calling the `MyLayer.__init__()` method as if it were a regular method, they are calling `MyLayer`. We don't want to generate documentation for the `__init__()` method as a standalone method that needs to be called directly, that would be confusing. We also don't need `__init__()` docstrings that always start with "Initializes a MyLayer class.", which is useless information. Leaving `__init__()` without a docstring is the best practice.
    - If constructor arguments are documented in `__init__`, it forces us to programmatically copy the `__init__` docstring when generating docs and concatenate it to the class docstring. This means that the `Args` section becomes the last thing in the docstring, which is bad.
- The order of information in a class docstring should be:
    - One-line description of the class, that gives initial context to the user. e.g. `Applies Dropout to the input.` Make sure the one-line description is useful. No `Intantiates an ObscureName class instance.`
    - Paragraph(s) of more detailed information that tells the user what the object is for and when they need to use it. e.g. `The Dropout layer randomly sets input units to 0 with a frequency of "rate" at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by "1/(1 - rate)" such that the sum over all inputs is unchanged. [...]`
    - If there is a reference paper, cite it here.
    - `Args` section.
    - If it's a layer that has arguments in `call`, the `Call arguments` section.
    - If it's a `Layer`, `Input shape` and `Output shape` sections.
    - Example(s).
    - Lastly, addendum. Information that isn't very important and that most users don't need, but that should be documented somewhere.
        - e.g. warnings about edge cases or compatibility issues.
        - e.g. pointers to further guides and tutorials.

### Error messages: a case study

The following would be a very poor error message:

```
AssertionError: '1 != 3'
```

In general, to validate user input, always use `ValueError` and avoid `assert`.

Also bad:

```
ValueError: 'Invalid target shape (600, 1).'
```

The following is better, but still not sufficient, because it does not tell the user what they passed, and does not quite say how to fix it:

```
ValueError: 'categorical_crossentropy requires target.shape[1] == classes'
```

Now, here's a good example, that says **what was passed**, **what was expected**, and **how to fix the issue**:

```
ValueError: '''You are passing a target array of shape (600, 1) while using as loss `categorical_crossentropy`.
`categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape (samples, classes).
If your targets are integer classes, you can convert them to the expected format via:

---
from keras.utils import to_categorical
y_binary = to_categorical(y_int)
---

Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.
```

---

## Implementation Best Practices (Multi-Backend & Symbolic Tensors)

Keras supports multiple backends (JAX, TensorFlow, PyTorch) and uses symbolic execution. Follow these patterns to ensure compatibility and robustness:

### Multi-Backend Compatibility

- **Prefer `backend.ops.convert_to_tensor` over backend-specific methods**: Use `backend.ops.convert_to_tensor(x)` instead of direct calls like `torch.as_tensor(x)`. This ensures proper handling of various input types, including Keras objects such as variables, and consistent detection of the dtype.
- **Backend-Agnostic Shape Handling**: Prefer using `backend.ops.shape(inputs)` (or a passed `backend_module.ops.shape(inputs)`) over the `.shape` property. This ensures consistency across JAX, TF, and Torch, especially for symbolic tensors.
- **Support Dynamic Dimensions**:
    - Use `len(shape)` or `ops.ndim(x)` to access shapes in a backend-agnostic way.
    - Use `isinstance(d, int)` to detect that a dimension is static and not dynamic. `None` is not the only representation for dynamic dimensions.
    - Use plain Python operators (e.g. `+`, `*`, `//`) to perform math on the dimensions of a shape, not Keras ops. This works seamlessly for static dimensions, symbolic dimensions (JAX, Torch) and tensor dimensions (TensorFlow). For instance, `math.prod(shape)` is the correct way to determine the size of an array.

### Backend-Agnostic Implementation for New Ops

Applies when a PR adds an op under `keras/src/ops/`. Every op must have 
an implementation written purely with other Keras ops, in addition to any
backend-specific fast path.

Flag the following in the diff:

- **No agnostic path**: the op is only implemented in `keras/src/backend/<backend>/ops/`
  and the `keras/src/ops/` function just forwards to backend-specific implementation.
  The op then breaks on any backend that does not implement it.
- **Logic not centralized**: the implementation is inlined in the public
  function or in `Operation.call()`, and the other one duplicates it. Both must
  delegate to a single private module-level `_my_op()` helper that owns the
  dispatch and the fallback.
- **Only one path tested**: the test is not parameterized with
  `@parameterized.named_parameters(named_product(BACKEND_AGNOSTIC_OPS))`.
- **Leaked test flag**: a test calls `_set_use_backend_agnostic_ops(...)`
  without immediately registering
  `self.addCleanup(backend.config._set_use_backend_agnostic_ops, False)`.
  A failing assertion then silently changes the behavior of every subsequent
  test in the process. Also flag new tests that reset the flag in a
  `try`/`finally` block and suggest `self.addCleanup` instead.
- **Public `ops` API in the fallback**: inside `keras/src/ops/`, the fallback
  must call backend ops directly — `backend.ops.<op>` for core ops (e.g.
  `backend.ops.cast`) and `backend.ops.<module>.<op>` otherwise (e.g.
  `backend.ops.numpy.sqrt`), or the dependency's `_<dep>()` helper if it has
  one. Flag calls to the public `ops.<op>` / `keras.ops` API. Never suggest
  replacing `backend.ops.*` with `ops.*`.

When flagging, cite "Backend-agnostic implementation for new ops" in
`CONTRIBUTING.md` and point to `vsplit`/`_vsplit` in `keras/src/ops/numpy.py`
as the reference implementation.

### Optimization & Numeric Stability

- **Division by Zero**: Use `ops.divide_no_nan` for mask weight calculations or any situation where a zero divisor is possible.
- **Arithmetic Masking**: Use `ops.where(mask, x, 0)` or `backend.ops.numpy.where(mask, x, 0)` instead of multiplication `x * mask` to clear values outside of a mask to save memory on intermediary values.

### API Design & Validation

- **Early Validation**: Perform axis canonicalization (using `canonicalize_axis` or `canonicalize_axes`) early to catch errors during graph building: typically in `compute_output_spec` or in the backend implementation of an op if the backend native op doesn't validate the axis. Validate axis uniqueness *after* canonicalization.

### Testing

- **Maximize Test Coverage**: Use specific test marks (like `@pytest.mark.requires_trainable_backend`) only when strictly necessary. Ensure training-related components are tested with a full `model.fit()` loop.
- **Verify Error Clarity**: Use `self.assertRaisesRegex` to verify that the error message contains the expected guidance.
- **Avoid Dynamic Shapes**: In layers, metrics, and losses avoid operations that have an output with a dynamic shape where possible as they are not compilable with JAX. This includes `ops.nonzero`, `ops.unique` without a `size`.

---

When performing code reviews on pull requests, you must strictly adhere to the following principles in addition to the API design guidelines above:

1. **Question the Necessity of Changes**: Do not assume that the pull request changes are strictly necessary. Critically review the proposed changes to ensure they add real value. Point out any code that is solving a non-existent problem or adding unnecessary complexity.
2. **Call out "AI Slop"**: Actively look for and identify "AI slop"—generic, overly verbose, or hallucinated code that lacks context or violates best practices. If you suspect the code is AI slop, explicitly call it out.
3. **Poke Holes in the Implementation**: Your goal is to critically test the logic. Actively search for and point out failing edge cases, race conditions, or unhandled exceptions in the implementation.
4. **Demand Robustness**: Do not accept fragile code. If the proposed code is not robust enough or lacks proper error handling, explicitly tell the author why the current approach is brittle and what must be done to reinforce it.
5. **Respect Existing Repo Patterns**: Before suggesting review comments (like asking users to add boilerplate or specific patterns), actively check for existing design patterns across the repository. Do not suggest adding useless code or structures that contradict or fall outside the established Keras repo coding style.
