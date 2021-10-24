from time import time

def timeIt(func: callable) -> float:
    start = time()
    func()
    return time() - start