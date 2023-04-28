

class AttributeDict(dict):
    def __getattribute__(self, item):
        if item not in self:
            return super().__getattribute__(item)

        item = self[item]
        if type(item) is dict:
            # only convert nested dicts if type is exactly dict
            return AttributeDict(item)

        return item

    def __setattr__(self, key, value):
        self[key] = value
