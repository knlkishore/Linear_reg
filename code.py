import numpy 

#functions
def compute_cost(X,Y,W):
    m,n = X.shape
    Cost = 0
    for i in range(m):
        Cost = Cost + (numpy.dot(X[i],W) - Y[i])**2
    Cost = Cost*(0.5/m)
    return Cost



#loading and saving training data
data = numpy.genfromtxt("data.csv" , delimiter= ",")
data = numpy.array(data)
x_data=numpy.c_[ numpy.ones(len(data)) , data[:,:-1] ]
y_data=data[:,-1]

#data correction
x_data[0][1] = 1


#parameter to minimize
n_loop = 5000
m,n = x_data.shape  #nnumber of examples, number of features)
w = numpy.zeros(n) #parameter

cost = numpy.ones(n_loop+1)

temp = numpy.zeros(n)
alpha = .00001


#gradient descend
tsum = 0

for count in range(1,n_loop):
    temp = w.copy()    
    for i in range(n):   
        for j in range(m):
            tsum = tsum + ( numpy.dot(w,x_data[j]) - y_data[j]  )*x_data[j,i]            
        temp[i] = temp[i] - (alpha/m)*tsum
    
    
    cost[count] = compute_cost(x_data,y_data,temp)

    
    cost[0] = cost[1]
    if i > 0 and cost[count] > cost[count-1]:
        break

    print(f"temp:{temp} || theta:{w} || Cost:{cost[count]}")
    w = temp.copy()
    count = count +1

#===========================================
print("last loop-----------------------------------------")
print(f"temp:{temp} || theta:{w} || Cost:{cost[count]}")

print(f"{count} iteration reached ")

"""
14	12.55555556
15	13.22222222
16	13.88888889
17	14.55555556
18	15.22222222
19	15.88888889
20	16.55555556
21	17.22222222
"""	


k = 7 
val = w[0] + k*w[1]
print(f" Predicted Value:{val}")