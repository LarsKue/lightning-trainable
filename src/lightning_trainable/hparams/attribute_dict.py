

class AttributeDict(dict):
    def __init__(self, **kwargs):
        for key in kwargs.keys():
            if type(kwargs[key]) is dict:
                kwargs[key] = AttributeDict(**kwargs[key])

        super().__init__(**kwargs)

    def __getattribute__(self, item):
        if item not in self:
            # not found, use default failure behavior
            return super().__getattribute__(item)

        return self[item]

    def __setattr__(self, key, value):
        if type(value) is dict:
            value = AttributeDict(**value)

        self[key] = value

    def copy(self):
        # copies of AttributeDicts should be AttributeDicts
        # see also https://github.com/LarsKue/lightning-trainable/issues/13
        return self.__class__(**super().copy())

    def as_dict(self):
        d = dict()
        for key, value in self.items():
            if isinstance(value, AttributeDict):
                value = value.as_dict()

            d[key] = value

        return d
