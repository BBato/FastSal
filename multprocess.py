from multiprocessing import Pool,TimeoutError 
import time

def func1():
    print('computing 1')
    time.sleep(2)
    return '1 ready.'

def func2():
    print('computing 2')
    time.sleep(2)
    return '2 ready.'


def pairTimedMethods(func1, func2, period):
    halfPeriod = period/2
    with Pool(processes=2) as pool:

        r2 = pool.apply_async(func2, ())
        time.sleep(halfPeriod)

        while True:
    
            r1 = pool.apply_async(func1, ())
            time.sleep(halfPeriod)
            print(r2.get())
            r2 = pool.apply_async(func2, ())
            time.sleep(halfPeriod)
            print(r1.get())

if __name__ == '__main__':
    pairTimedMethods(func1, func2, 2)