import time


class Timer:
    def __init__(self, name, results):
        self.name = name
        self.results = results
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.perf_counter() - self.start_time
        self.results[self.name] = elapsed_time
        print(f"[{self.name}]: {elapsed_time:4f} segundos")
