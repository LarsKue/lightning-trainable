
class Choice:
    """
    One of several choices
    Usage:
    class MyHParams(HParams):
        value: Choice("asdf", 3, 7.5)

    hparams = MyHParams(value="asdf")
    assert hparams.value == "asdf"
    """
    def __init__(self, *choices):
        self.choices = choices

    def __instancecheck__(self, instance):
        return instance in self.choices
