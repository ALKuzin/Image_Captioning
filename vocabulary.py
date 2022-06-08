class Vocabulary:

    def __init__(self):
        self.itos = {}

    def __len__(self):
        return len(self.itos)

    def load_vocab(self):
        with open('settings/itos.txt') as itos:
            for i in itos.readlines():
                key, val = i.strip().split('|', 1)
                self.itos[int(key)] = val
