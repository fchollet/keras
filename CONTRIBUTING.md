Keras 3 is a high-velocity open-source project. We welcome contributions!

Contributions can be made in a variety of ways, including coding, enriching documentation, refining docstrings, and providing code examples.

Please review our
[AI-Assisted Contribution Policy](#ai-assisted-contribution-policy).

## Current items open for contributions
At [this link](https://github.com/keras-team/keras/issues/18442), you'll find a list of items where your help is needed!


## How to contribute code

Follow these steps to submit your code contribution.

### Step 1. Open an issue

Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validate the proposed changes. Unsolicited PRs that attempt to fix complex
issues without prior discussion in a GitHub issue may be closed.

If the changes are minor (simple bug fix or documentation fix), then feel free
to open a Pull Request (PR) without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to setup a
development environment and run the unit tests. This is covered in the section
"Setup environment".

### Step 3. Create a pull request

Once the change is ready, open a pull request from your branch in your fork to
the master branch in [keras-team/keras](https://github.com/keras-team/keras).

### Step 4. Sign the Contributor License Agreement

After creating the pull request, the `cla/google` check will be performed and,
if you haven't signed the Contributor License Agreement (CLA), it will fail with
instructions on how to do so. Please follow the instructions to sign the CLA and
the check will pass.

![CLA signed](https://github.com/keras-team/keras/assets/1091026/71c26353-e3b5-4135-8bae-64693c717775)


### Step 5. Code review

If the tests fail, look into the error messages and try to fix them.

![CI tests](https://github.com/keras-team/keras/assets/1091026/6f6c17ef-6bd7-4e95-9fbc-1906cde37380)

A reviewer will review the pull request and provide comments. There may be
several rounds of comments and code changes before the pull request gets
approved by the reviewer.

![Approval from reviewer](https://github.com/keras-team/keras/assets/1091026/8d28f74c-21e9-4146-b0ff-62d649a552a8)

### Step 6. Merging

Once the pull request is approved, a `ready to pull` tag will be added to the
pull request. A team member will take care of the merging.

![Ready to pull and merged](https://github.com/keras-team/keras/assets/1091026/c3908345-d7ae-44ee-a428-01f3b448b46b)

Here is an [example pull request](https://github.com/keras-team/keras/pull/18848)
for your reference.

## Setup environment

We provide two ways of setting up a development environment. One is to use a
dev container, and the other one is to set up a local environment by installing
the dev tools needed.

### Option 1: GitHub Codespace or dev container

We support GitHub Codespaces, Visual Studio Code dev containers and JetBrain dev
containers. Please see the
[Dev container documentation](https://github.com/keras-team/keras/tree/master/.devcontainer).

### Option 2: Set up a local environment

To set up your local dev environment, you will need the following tools.

1.  [git](https://github.com/) for code repository management.
2.  [python](https://www.python.org/) to build and code in Keras.

The following commands check the tools above are successfully installed. Note
that Keras requires at least Python 3.10 to run.

```shell
git --version
python --version
```

Clone your forked repo to your local machine. Go to the cloned directory to
install the dependencies.

```shell
git clone https://github.com/YOUR_GITHUB_USERNAME/keras.git
cd keras
pip install -r requirements.txt
```

You then need to configure the backend to use, see the
[Configuring your backend](https://github.com/keras-team/keras/blob/master/README.md#configuring-your-backend)
section of the README.

You can also add GPU support to your environment, see the
[Adding GPU support](https://github.com/keras-team/keras/blob/master/README.md#adding-gpu-support)
section of the README.

## Generating public API and formatting the code

For the first time you are setting up the repo, please run `pre-commit install`.
Note that this needs to be done only once at the beginning.

Now, whenever you run `git commit -m "<message>"`, three things are
automatically done:

- Public API generation
- Code formatting
- Code linting

If there's any error, the commit will not go through. Please fix the error (
most of the times, the error is fixed automatically by the formatter/linter) and
re-run the following:

```
git add .
git commit -m "<message>" # This will not get logged as a duplicate commit.
```

In case you want to run the above manually on all files, you can do the
following:

```
pre-commit run --all-files
```

KerasHub uses [Ruff](https://docs.astral.sh/ruff/) to format the code.

### Docstrings

We do not have an automated way to check docstring style, so if you write
or edit any docstring, please make sure to check them manually.
Keras docstrings follow the conventions below:

A **class docstring** may contain the following items:

* A one-line description of the class.
* Paragraph(s) of more detailed information.
* Optional `Examples` section.
* `Args` section for arguments in `__init__()`.
* If it's a layer:
    * `Call arguments` section for arguments in `Layer.call()`.
    * `Returns` section for the return values of `Layer.call()`.
    * Optional `Raises` section for possible errors.

You can check out `MultiHeadAttention` as an example
[(link)](https://github.com/keras-team/keras/blob/v3.0.0/keras/layers/attention/multi_head_attention.py#L20).

A **function docstring** may contain the following items:

* One-line description of the function.
* Paragraph(s) of more detailed information.
* Optional `Examples` section.
* `Args` section for the function arguments.
* `Returns` section for the return values.
* Optional `Raises` section for possible errors.

You can check out `text_dataset_from_directory` as an example
[(link)](https://github.com/keras-team/keras/blob/v3.0.0/keras/utils/text_dataset_utils.py#L27).

## Run tests

We use [pytest](https://pytest.org/) to run the tests.

### Run a test file

To run the tests in `keras/src/losses/losses_test.py`, use the following command
at the root directory of the repo.

```shell
pytest keras/src/losses/losses_test.py
```

### Run a single test case

You can specify a single test class to run within a file.

```shell
pytest keras/src/losses/losses_test.py::MeanSquaredErrorTest
```

You can also specify a single test method to run within a class.

```shell
pytest keras/src/losses/losses_test.py::MeanSquaredErrorTest::test_sample_weighted
```

### Run all tests

You can run all the tests locally by running the following command in the repo
root directory.

```shell
pytest keras
```

Note that you can skip the Keras applications tests using the
`SKIP_APPLICATIONS_TESTS` environment variable. This will cut down the testing
time significantly.

```shell
SKIP_APPLICATIONS_TESTS=True pytest keras
```

To run all tests using a different backend, you can simply specify it on the
command line.

```shell
KERAS_BACKEND=jax SKIP_APPLICATIONS_TESTS=True pytest keras
```
## Backend-agnostic implementation for new ops

Any new op that is implemented must include a backend-agnostic implementation.
A backend-agnostic implementation is the version of the op written in
`keras/src/ops/` using only other Keras ops, so that it works on every backend
without requiring per-backend code. This guarantees that an op is
immediately available on all backends, including ones that lack a backend-specific
implementation.

### The pattern

Put the dispatching and fallback logic in a private module-level helper named
`_op()` (where `op` is the name of the op), and have both the public function
and its `Operation` subclass call it.

```python
class MyOp(Operation):
    def call(self, x):
        return _my_op(x)

    ...


def my_op(x):
    if any_symbolic_tensors((x,)):
        return MyOp().symbolic_call(x)
    return _my_op(x)


def _my_op(x):
    if not config._use_backend_agnostic_ops() and hasattr(
        backend.ops.numpy, "my_op"
    ):
        return backend.ops.numpy.my_op(x)
    x = backend.ops.convert_to_tensor(x)
    ...
    return res
```

Inside `keras/src/ops/`, the fallback should call backend ops directly:
`backend.ops.<op>` for core ops (e.g. `backend.ops.cast`) and
`backend.ops.<module>.<op>` for everything else (e.g.
`backend.ops.numpy.where`). Don't call the public `keras.ops` API. If a
dependency has its own `_<dep>()` helper, call that instead.

[Here](https://github.com/keras-team/keras/blob/5edcf00a9e818838988c8c0cf45a79e279851803/keras/src/ops/numpy.py#L8606-L8664)
is a minimal version of `vsplit` from `keras/src/ops/numpy.py`.

### Testing both paths

Both code paths must be tested. Parameterize the test with the
`BACKEND_AGNOSTIC_OPS` constant and register a reset of the flag with
`self.addCleanup` immediately after setting it.

```python
@parameterized.named_parameters(named_product(BACKEND_AGNOSTIC_OPS))
def test_my_op(self, backend_agnostic_ops):
    backend.config._set_use_backend_agnostic_ops(backend_agnostic_ops)
    self.addCleanup(backend.config._set_use_backend_agnostic_ops, False)
    x = np.array([1.0, 2.0, 3.0])
    self.assertAllClose(knp.my_op(x), np.my_op(x))
    self.assertAllClose(knp.MyOp()(x), np.my_op(x))
    ...
```

The cleanup runs after the test finishes, even if an assertion fails, so the
flag never leaks into subsequent tests. Prefer `self.addCleanup` over a
`try`/`finally` block: it keeps the test body unindented.

[Here](https://github.com/keras-team/keras/blob/f3680413cbfb503b43ecb64818ccae1df5b7096c/keras/src/ops/numpy_test.py#L7454-L7491)
is an example of testing both code paths for `vsplit`.

You can also test the fallback across an entire test run by setting the
`KERAS_USE_BACKEND_AGNOSTIC_OPS` environment variable:

```shell
KERAS_USE_BACKEND_AGNOSTIC_OPS=1 pytest keras/src/ops/numpy_test.py
```

## GitHub Actions Security Validation

Pull requests modifying GitHub Actions workflows are automatically validated using Zizmor.
Before requesting review:

- Resolve all Zizmor findings whenever possible.
- Run Zizmor locally when modifying workflow files (e.g., using `uvx zizmor .github/` or `pipx run zizmor .github/`).
- Use `# zizmor: ignore[rule-name]` only for verified false positives.
  - Examples: `# zizmor: ignore[cache-poisoning]`, `# zizmor: ignore[dangerous-triggers]`.
  - For a full list of rules, see the [Zizmor Rules Documentation](https://docs.zizmor.sh/audits/).
- Every suppression must include a clear justification explaining why the finding is safe.

## AI-Assisted Contribution Policy

The Keras project relies on a vibrant, collaborative open-source community. As
an organization at the forefront of machine learning, we recognize and support
the use of AI-assisted coding tools to enhance developer productivity.

The ultimate responsibility for any code contributed to Keras rests entirely
with the human author. To maintain the high quality, security, and architectural
integrity of the Keras codebase, and to respect the time of our maintainers, we
require all contributors to adhere to the following policy.

### 1. Disclosure Requirements

If you used an AI coding agent in any capacity in the process of creating the
pull request, you must disclose this in the PR description. This provides
necessary context for the reviewers. By opening a pull request which includes AI
generated code, you accept all responsibility for the code in question and
guarantee that the content of the pull request complies with the terms of Keras’
open source license and intellectual property policies.

### 2. Acceptable vs. Unacceptable Use

#### Acceptable Use:

- Using AI tools while writing code you understand.
- Generating boilerplate code, documentation drafts, or unit test templates that
  you manually review, refine, and test.
- Using LLMs to help you understand a complex bug or explain a piece of the
  existing Keras codebase locally.

#### Unacceptable Use:

- **Zero-Review Agent PRs:** Allowing an autonomous AI agent to read an issue,
  generate a patch, and open a Pull Request without comprehensive manual review
  and testing by you.
- **Blind Copy-Pasting:** Submitting patches generated by an LLM where you do
  not fully grasp the underlying mechanics of the fix.
- **AI-Generated Code Review Responses:** Using an LLM to automatically generate
  replies to maintainer feedback. Code reviews require human-to-human
  communication. If a maintainer asks an architectural question, you must answer
  it yourself.

### 3. Enforcement and PR Closure

To keep the project moving efficiently, the Keras maintainer team reserves the
right to enforce this policy strictly:

- **Unaddressed Feedback:** If a maintainer leaves structural or architectural
  feedback and the PR is abandoned, or the responses demonstrate a lack of
  understanding of the submitted code, the PR will be closed.
- **Policy Violation:** Repeated submission of low-effort, poorly understood
  AI-generated code will result in PR closure and potential restriction from
  contributing to the repository.

### Acknowledgment

By submitting a Pull Request to Keras, you confirm that you have read this
policy, understand the code you are submitting, and take full responsibility for
its accuracy and integration into the codebase.
