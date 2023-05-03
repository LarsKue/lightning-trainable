
from typing import Union


class HParamType:
    def __or__(self, other):
        return Union[type(self), other]

    def __ror__(self, other):
        return Union[other, type(self)]

    def __instancecheck__(self, instance):
        raise NotImplementedError
