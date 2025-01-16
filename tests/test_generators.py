from datafast.generators import random_string


def test_random_string_length():
    result = random_string(15)
    assert len(result) == 15


def test_random_string_default_length():
    result = random_string()
    assert len(result) == 10
