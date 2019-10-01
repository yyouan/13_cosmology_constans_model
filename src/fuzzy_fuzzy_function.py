import numpy as np
from scipy.stats import truncnorm

class fuzzy:
    def __init__(self, normalized_array ,rrange ,partion):
        self.normalized_array= normalized_array
        self.range = rrange
        self.partion = partion

    def cut(self):
        print('===== fuzzy cut... =============')
        np.random.seed(self.partion)
        def k_norm(i,j,k):
            a = truncnorm.pdf(
                                (2*(k/self.partion)*5.5-5.5),-3.0,3.0,
                                loc= self.normalized_array[int(i)][int(j)],
                                scale = self.range,
                            )
            if a==0.0:
                return 0.01*np.random.rand()
            else:
                return a

        return np.fromfunction(
                        np.vectorize(k_norm),
                        (self.normalized_array.shape[0],self.normalized_array.shape[1],self.partion),
                        dtype='float32')
    
    def fuzzy_linspace(self):
        return np.linspace(-5.5,5.5,self.partion)

if __name__ == '__main__':
    array = np.array([[0.0,1.0,2.0,3.0,4.0,5.0]])
    print(fuzzy(array,1.0,10).cut())  

'''

we notice 'scale' isn't mattter in terms of possiblility's distribution
so we do not multiply distribuiton density value by the x range under it
to let diffenrent distributions' index value has similar scale ,so their
sum won't be at the same scale....

'''