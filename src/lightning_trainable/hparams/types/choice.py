

class ChoiceMeta(type):
    def __instancecheck__(self, instance):
        return instance in self.choices

    def __call__(cls, *choices):
        # dynamically construct a new class with the given choices
        name = f"Choice{choices!r}"
        bases = (Choice,)
        namespace = {"choices": choices}
        return type(name, bases, namespace)


class Choice(metaclass=ChoiceMeta):
    """
    One of several choices
    Usage:
    class MyHParams(HParams):
        value: Choice("asdf", 3, 7.5)

    hparams = MyHParams(value="asdf")
    assert hparams.value == "asdf"
    """
    pass
