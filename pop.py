import bisect
import math
from spacepy import poppy
import matplotlib
import matplotlib.pyplot as plt
import numpy
#Create the s e r i e s

gauss = lambda x : math.exp(-(float(x-20) ** 2)/(2 * 10 ** 2 ))/(10 * math.sqrt(2 * math.pi))
lags = range(-60 * 6 , 60 * 6 + 1 , 10) #minutes , up to qua r t e r day
numpy.random.seed(1337)
series_1 = numpy.random.randint (60 * -15 * 6 , 60 * 105 * 6 + 1 , [ 60 * 6 ] )
series_1 .sort( )
series_1 = numpy.array( series_1 , dtype='float64')
randvals = numpy.random.rand(60 * 90 * 6 + 1)
series_2 = numpy.fromiter((
    i for i in range(0, 60 * 90 * 6 + 1) if gauss(i - series_1[ bisect.bisect_right(series_1, i )- 1 ] ) +
    gauss( i - series_1 [bisect.bisect_left(series_1 , i ) ])
    > 0.25 * randvals[i]), numpy.float64 , count=-1)
#Create a PPro o b j e c t from the s e r i e s
pop = poppy.PPro(series_1 , series_2 , lags = lags , winhalf =12.0)
pop.assoc() #Perform a s s o c i a t i o n a n a l y s i s
#Figure 2
pop.plot(norm=False)
pop.aa_ci(95 , n_boots=4000) #Generate conf idenc e i n t e r v a l s
#Do the same f o r the s e r i e s in the r e v e r s e order
poprev = poppy.PPro (series_2 , series_1 , lags = lags[-1: : -1] , winhalf =12.0)
poprev.assoc()
poprev.aa_ci(95 , n_boots=4000)
poprev.lags = lags
#Figure 3
poppy.plot_two_ppro(pop , poprev , ratio =1.0)
plt.show()
#Set up f o r window s ear ch al gor i thm
pop_2d = poppy.PPro ( series_1 , series_2 , lags= lags , winhalf =12.0)
windows = numpy.array(range(30) , dtype= 'float64' )
( low , high , percentile ) = pop_2d.assoc_mult(windows , n_boots=10000 , seed=1337)
#Figure 5
pop_2d.plot_mult(windows, percentile , min=95.0)
plt.show()
#Figure asymptot i c a s s o c i a t i o n f o r a l l windows
13
asymptotes = numpy.empty([30] , dtype= 'float64' )
for i in range ( len(windows) ) :
    junk = poppy.PPro(series_1 , series_2 , lags = lags , winhalf = windows[i])
    junk.assoc(h=windows[i])
    asymptotes[i] = junk.asympt_assoc
#Rat io of bot tom of c . i . to asymptote
ratios = numpy.empty(low.shape , dtype= 'float64')
for i in range (len(windows)):
    ratios[ i , : ] = low [ i , : ] / asymptotes [ i ]
#Figure 6
pop_2d.plot_mult(windows, ratios, min=1.4)
plt.show ( )