"""Plain-assert checks for the pure-logic parts of the agent (no audio, no models).

Run: uv run python test_logic.py
Each test_* function raises AssertionError if the logic is broken.
"""
import config


def test_config_profiles_have_same_keys():
    # If a key exists in [pc] but not in [pi] the Pi would crash at startup with
    # AttributeError. Loading both and comparing their keys catches that on the PC.
    pc, pi = config.load_config("pc"), config.load_config("pi")
    assert set(vars(pc)) == set(vars(pi)), set(vars(pc)) ^ set(vars(pi))
    assert pc.profile == "pc" and pi.profile == "pi"


if __name__ == "__main__":
    # Collect every function whose name starts with test_ and run it.
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} checks passed")
