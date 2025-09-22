from common.exceptions import TaskTimeoutError, run_with_timeout


def test_run_with_timeout_ok():
    def f(x):
        return x + 1

    assert run_with_timeout(f, 0.2, 1) == 2


def test_run_with_timeout_timeout():
    import time

    def slow(x):
        time.sleep(0.2)
        return x

    try:
        run_with_timeout(slow, 0.05, 1)
        raise AssertionError("expected timeout")
    except TaskTimeoutError:
        assert True
