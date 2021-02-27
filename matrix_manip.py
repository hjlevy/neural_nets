import numpy as np 

a = np.random.rand(5,10)
b = np.random.rand(7,10)
print(np.shape(a))
N = np.shape(a)[0]
num_a = len(a)
 
print('a = ')
print(a)

# print('a2: = ')
# # print(a[0:1])

# print(np.arange(num_a))
# print(num_a)
# i = 0
# print(np.concatenate((a[0:i],a[i+1:5]),axis=0))

# testing tile function
# logk= -np.amax(a, axis =1, keepdims = True)
# print(np.shape(logk))
# print(logk)
# print('ones')
# print(np.tile(logk,(1,10)))

# y = np.random.randint(0,10,N)
# ind = [:,y]
# print(N)
# print(ind)
# print(a[ind])

# N = a.shape[0]
# batch_size = 3
# index = np.random.choice(N,batch_size)
# print(index)
# print(a[index])
b = np.sum(a,axis=1)
print(b)
print(np.shape(b))