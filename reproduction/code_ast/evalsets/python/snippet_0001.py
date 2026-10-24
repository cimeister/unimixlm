MUTATION = '''mutation {{
  {mutation}
}}'''


def _verify_additional_type(additionaltype):
    """Check that the input to additionaltype is a list of strings.
    If it is empty, raise ValueError
    If it is a string, convert it to a list of strings."""
    if additionaltype is None:
        return None

    if isinstance(additionaltype, str):
        additionaltype = [additionaltype]
    if len(additionaltype) == 0:
        raise ValueError("additionaltype must be a non-empty list")
    return additionaltype