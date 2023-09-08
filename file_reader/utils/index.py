class Indexer:
    def __init__(self, value=0):
        self.value = value

    def increase(self):
        self.value += 1


def indexer(data, index=None):
    if index is None:
        index = Indexer()

    def _index(data_, idx):

        if isinstance(data_, list):
            for d in data_:
                _index(d, idx=idx)
        elif isinstance(data_, dict):
            data_['index'] = idx.value
            idx.increase()
            _index(data_['children'], idx)

    _index(data, index)
    return data
