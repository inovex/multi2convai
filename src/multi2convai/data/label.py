class Label:
    """Data type for pipelines return value

    Args:
        string (str): label string
        ratio (float): label ratio
    """

    def __init__(self, string: str, ratio: float):
        self.string = string
        self.ratio = ratio

    def __str__(self):
        return self.string

    def __repr__(self):
        return f"Label(string='{self.string}', ratio='{self.ratio}')"
