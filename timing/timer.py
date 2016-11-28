import time

class Timer:
    def __init__(self):
        self.start_t = 0.0
        self.stop_t = 0.0
        self.last_time = 0.0

    def start_timer(self):
        self.last_time = self.get_time()
        self.start_t = time.time()

    def stop_timer(self):
        self.stop_t = time.time()

    def get_time(self):
        return self.stop_t - self.start_t

    def reset_timer(self):
        self.last_time = self.get_time()

class RunTest(Timer, object):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.delay = 4

    def run_test(self):
        time.sleep(self.delay)

if __name__ == "__main__":

    rt = RunTest()

    rt.start()

    rt.run_test()

    rt.stop()

    print(rt.get_time())
