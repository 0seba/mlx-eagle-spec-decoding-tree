import time
import mlx.core as mx


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


def round_cast(x, *args, **kwargs):
    return mx.round(x.astype(mx.float32), *args, **kwargs)


def round_to_list(x, decimals):
    return [round(v, decimals) for v in x.tolist()]


def print_2d(x, decimals):
    # Print column numbers
    col_width = decimals + 3  # width for each number (1 for sign, decimals, and decimal point)
    print("     " + "  ".join(f"{i:{col_width}d}" for i in range(len(x[0]))))
    # Print rows with row numbers
    for i, row in enumerate(x):
        print(f"{i:3d}  " + "  ".join("{: {width}.{prec}f}".format(v, width=col_width, prec=decimals) for v in row))


def washout():
    mx.synchronize()
    mx.eval(
        mx.full((10 * 1024**2,), 1.0) * mx.full(10 * 1024**2, 2.0)
    )  # clear cache, each of this is 40MB
    mx.synchronize()
    mx.metal.clear_cache()
    mx.synchronize()


def benchmark(f, x):
    times = []
    mx.synchronize()
    for i in range(30):
        tic = time.perf_counter()
        # mx.eval(qmv(x, w, s, b, 64))
        mx.eval(f(x))
        toc = time.perf_counter()
        times.append(toc - tic)
        washout()

    times = times[10:]
    # return mean and median
    return (sum(times) / len(times), sorted(times)[len(times) // 2])


def bench_print(f, name, x):
    mean, median = benchmark(f, x)
    print(f"Average time {name}: {mean:.5f} ms")
    print(f"Median time {name}: {median:.5f} ms")
