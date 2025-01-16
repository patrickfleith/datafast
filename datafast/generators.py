import random
import string


def random_string(length: int = 10) -> str:
    """
    Generate a random string of lowercase letters and digits.

    :param length: The length of the generated string.
    :return: A random string of given length.
    """
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))
