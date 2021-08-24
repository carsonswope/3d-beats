import time

class ProfileTimer:
    def __init__(self):
        self.events = [] # (name|None, start time)

    def record(self, name):
        self.events.append((name, time.perf_counter()))

    def render(self):
        self.record(None)
        e = []
        max_chars = max([len(n) if n else 0 for n, _ in self.events])
        for i in range(len(self.events) - 1):
            name, start_time = self.events[i]
            _, end_time = self.events[i+1]
            t = (end_time - start_time) * 1000
            e.append(f'{name.ljust(max_chars)}: {"%.2f" % t} ms')

        total_time = (self.events[-1][1] - self.events[0][1]) * 1000
        e.append(f'{"total time".ljust(max_chars)}: {"%.2f" % total_time}')

        self.clear()
        return e

    def clear(self):
        self.events.clear()
