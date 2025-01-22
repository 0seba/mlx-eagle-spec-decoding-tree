import time


class Performer:
    def __init__(self, n_warmup):
        self.n_warmup = n_warmup
        self.counters = {}
        self.timers = {}

    def tic(self, name):
        if name not in self.counters:
            self.counters[name] = []
        self.timers[name] = time.perf_counter()

    def toc(self, name):
        self.counters[name].append(time.perf_counter() - self.timers[name])

    def get_average(self, name):
        # at least 1 element\
        if len(self.counters[name]) < self.n_warmup:
            n = len(self.counters[name]) - 1
        else:
            n = self.n_warmup
        return sum(self.counters[name][n:]) / (len(self.counters[name]) - n)

    def print_all_averages_ms(self, rounding_digits=3):
        for name in self.counters:
            print(f"{name}: {self.get_average(name) * 1000:.{rounding_digits}f} ms")