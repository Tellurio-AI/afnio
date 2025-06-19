import os

import pytest

from afnio._variable import Variable
from afnio.autodiff.basic_ops import Add, Split
from afnio.autodiff.evaluator import DeterministicEvaluator
from afnio.autodiff.grad_mode import no_grad
from afnio.tellurio import login
from afnio.tellurio.run import init


@pytest.fixture(autouse=True)
def login_and_ensure_default_run():
    """
    Test the login function with real HTTP and WebSocket connections and
    ensure a default Run exists and is set as active before tests.
    """
    # Log in to the Tellurio service using the API key
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    login(api_key=api_key)

    # Use your test org/project names from env or defaults
    namespace_slug = os.getenv("TEST_ORG_SLUG", "tellurio-test")
    project_display_name = os.getenv("TEST_PROJECT", "Test Project")
    run = init(namespace_slug, project_display_name)
    return run


@pytest.fixture
def variables():
    x = Variable(data="abc", role="first input", requires_grad=True)
    y = Variable(data="def", role="second input", requires_grad=False)
    return x, y


class TestFunctionSync:
    """
    Tests for forward and backward passes of Function operations in afnio.autodiff.
    """

    def test_forward_add(self, variables):
        """
        Test the Add function's forward pass with two Variable inputs.
        """
        x, y = variables
        result = Add.apply(x, y)
        assert isinstance(result, Variable)
        assert result.data == "abcdef"
        assert result.role == "first input and second input"
        assert result.requires_grad is True
        assert "<AddBackward" in str(result.grad_fn)
        assert "<afnio.autodiff.function.AddBackward" in repr(result.grad_fn)
        assert len(result.grad_fn.next_functions) == 2
        assert "<AccumulateGrad" in str(result.grad_fn.next_functions[0].node)
        result.grad_fn.next_functions[0].output_nr == 0
        assert "None" in str(result.grad_fn.next_functions[1].node)
        result.grad_fn.next_functions[1].output_nr == 0
        assert result.is_leaf is False

    def test_forward_add_no_grad(self, variables):
        """
        Test the Add function's forward pass with two Variable inputs
        and using the `no_grad()` context manager.
        """
        x, y = variables
        with no_grad():
            result = Add.apply(x, y)
        assert isinstance(result, Variable)
        assert result.data == "abcdef"
        assert result.role == "first input and second input"
        assert result.requires_grad is False
        assert result.grad_fn is None
        assert result.is_leaf is True

    def test_forward_split(self, variables):
        """
        Test the Split function's forward pass with single Variable input.
        """
        x, _ = variables
        x.data = "a b c"
        result = Split.apply(x, sep=" ")
        assert isinstance(result, tuple)
        assert all(isinstance(v, Variable) for v in result)
        assert [v.data for v in result] == ["a", "b", "c"]
        expected_roles = [f"split part {i} of first input" for i in range(len(result))]
        for i, v in enumerate(result):
            assert v.role == expected_roles[i]
            assert v.requires_grad is True
            assert "<SplitBackward" in str(result[i].grad_fn)
            assert "<afnio.autodiff.function.SplitBackward" in repr(result[i].grad_fn)
            assert len(v.grad_fn.next_functions) == 1
            assert "<AccumulateGrad" in str(v.grad_fn.next_functions[0].node)
            assert v.grad_fn.next_functions[0].output_nr == 0

    def test_forward_deterministic_evaluator(self):
        """
        Test the DeterministicEvaluator function's forward pass with Callable input.
        """

        def exact_match_fn(pred: str, tgt: str) -> int:
            return 1 if pred == tgt else 0

        fn_purpose = "exact match"
        prediction = Variable(data="green", role="color prediction", requires_grad=True)
        target = Variable(data="red", role="expected color")
        result = DeterministicEvaluator.apply(
            prediction, target, exact_match_fn, fn_purpose, None, None, None
        )
        score, explanation = result
        assert isinstance(result, tuple)
        assert isinstance(score, Variable)
        assert isinstance(explanation, Variable)

        # Check score and explanation attributes
        assert score.data == 0
        assert score.role == "Evaluation result score of color prediction"
        assert score.requires_grad is True
        assert explanation.data == (
            "The evaluation function, designed for 'exact match', "
            "compared the <DATA> field of the predicted variable ('green') with "
            "the <DATA> field of the target variable ('red'), "
            "resulting in a score: 0."
        )
        assert explanation.role == "Evaluation result explanation of color prediction"
        assert explanation.requires_grad is True

        # Check grad_fn for score and explanation
        for var in (score, explanation):
            assert "<DeterministicEvaluatorBackward" in str(var.grad_fn)
            assert len(var.grad_fn.next_functions) == 2
            assert "<AccumulateGrad" in str(var.grad_fn.next_functions[0].node)
            assert var.grad_fn.next_functions[0].output_nr == 0
            assert "None" in str(var.grad_fn.next_functions[1].node)
            assert var.grad_fn.next_functions[1].output_nr == 0

    def test_backward_add(self, variables):
        """
        Test the Add function's backward pass.
        """
        x, y = variables
        result = Add.apply(x, y)
        assert isinstance(result, Variable)
        assert result.data == "abcdef"

        gradient = Variable(data="MY_FEEDBACK", role="add gradient")
        result.backward(gradient)

        assert len(x.grad) == 1
        assert y.grad == []  # requires_grad=False, so no gradient
        assert x.grad[0].data == (
            "Here is the combined feedback we got for this specific "
            "first input and other variables: MY_FEEDBACK"
        )
        assert x.grad[0].role == "feedback to first input"

    def test_backward_clear_pending_grad(self):
        """
        Test that _pending_grad is set during backward
        and cleared after 'clear_backward'.

        DAG:
                b → c → d
              ↗           ↘
            a  → → → → → → e
        """
        a = Variable(data="abc_", role="first input", requires_grad=True)
        b = a + Variable(data="def_", role="second input")
        c = b + Variable(data="ghi_", role="third input")
        d = c + Variable(data="jkl_", role="fourth input")
        e = Add.apply(a, d)

        gradient = Variable(data="MY_FEEDBACK", role="add gradient")
        e.backward(gradient)

        # The variable should have _pending_grad set before clear_backward
        assert a._pending_grad is True

        # We are only able to read `a.grad` when we exit `_wait_for_pending()`
        # and at that point, _pending_grad should be True
        assert len(a.grad) == 2
        assert a._pending_grad is False
