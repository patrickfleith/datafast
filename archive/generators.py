import random
import string


def random_string(length: int = 10) -> str:
    """
    Generates a random string of lowercase letters and digits.

    Args:
        length (int): The length of the generated string.

    Returns:
        str: A random string of the given length.
    """
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))

