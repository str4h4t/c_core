import correlation.DTW.dtw_calculator_2 as calc
#import multiprocessing as mp

def it(size):
    type = ["sakoechiba", "itakura", "slantedband"]

    for t in type:
        print("Type: " + t)
        print("Window Size: " + str(size))

        calc.calc(t,size)

if __name__ == "__main__":
    size = [6]
    #pool = mp.Pool(mp.cpu_count())
    #pool.map(it, [i for i in size])
    [it(i) for i in size]