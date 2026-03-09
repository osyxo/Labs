import numpy as np
import matplotlib.pyplot as plt
import csv

# ---------- зчитування CSV ----------
def read_data(filename):

    x=[]
    y=[]

    with open(filename,'r') as file:
        reader=csv.DictReader(file)

        for row in reader:
            x.append(float(list(row.values())[0]))
            y.append(float(list(row.values())[1]))

    return np.array(x),np.array(y)


# ---------- розділені різниці ----------
def divided_diff(x,y):

    n=len(y)
    coef=np.copy(y)

    for j in range(1,n):
        coef[j:n]=(coef[j:n]-coef[j-1:n-1])/(x[j:n]-x[0:n-j])

    return coef


# ---------- поліном Ньютона ----------
def newton_poly(coef,x_data,x):

    n=len(coef)
    result=coef[n-1]

    for k in range(1,n):
        result=coef[n-k-1]+(x-x_data[n-k-1])*result

    return result


# ---------- факторіальний поліном ----------
def factorial_poly(x,y,xp):

    n=len(x)

    h=x[1]-x[0]
    s=(xp-x[0])/h

    diff=[y.copy()]

    for i in range(1,n):
        diff.append([diff[i-1][j+1]-diff[i-1][j] for j in range(n-i)])

    result=y[0]
    fact=1
    s_term=1

    for i in range(1,n):
        s_term*= (s-(i-1))
        fact*=i
        result+= (s_term/fact)*diff[i][0]

    return result


# ---------- створення додаткових вузлів ----------
def generate_nodes(x,y,n):

    x_new=np.linspace(min(x),max(x),n)
    y_new=np.interp(x_new,x,y)

    return x_new,y_new


# ---------- головна програма ----------

x,y=read_data("data.csv")

nodes=[5,10,20]

plt.figure(figsize=(12,5))

# ---------- графік Ньютона ----------
plt.subplot(1,2,1)

for n in nodes:

    xn,yn=generate_nodes(x,y,n)

    coef=divided_diff(xn,yn)

    xs=np.linspace(min(x),max(x),200)
    ys=[newton_poly(coef,xn,i) for i in xs]

    plt.plot(xs,ys,label=f"Newton n={n}")

plt.scatter(x,y,color="black")
plt.title("Інтерполяція Ньютона")
plt.xlabel("Tasks")
plt.ylabel("Cost")
plt.legend()
plt.grid()


# ---------- графік факторіального ----------
plt.subplot(1,2,2)

for n in nodes:

    xn,yn=generate_nodes(x,y,n)

    xs=np.linspace(min(x),max(x),200)
    ys=[factorial_poly(xn,yn,i) for i in xs]

    plt.plot(xs,ys,label=f"Factorial n={n}")

plt.scatter(x,y,color="black")
plt.title("Факторіальний поліном")
plt.xlabel("Tasks")
plt.ylabel("Cost")
plt.legend()
plt.grid()

plt.show()


# ---------- графік похибок ----------

x_test=np.linspace(min(x),max(x), 100)

errors_newton=[]
errors_fact=[]

for n in nodes:

    xn,yn=generate_nodes(x,y,n)

    coef=divided_diff(xn,yn)

    err_n=0
    err_f=0

    for xt in x_test:

        true=np.interp(xt,x,y)

        pred_n=newton_poly(coef,xn,xt)
        pred_f=factorial_poly(xn,yn,xt)

        err_n+=abs(true-pred_n)
        err_f+=abs(true-pred_f)

    errors_newton.append(err_n)
    errors_fact.append(err_f)


plt.figure()

plt.subplot(1,2,1)
plt.plot(nodes,errors_newton,"o-")
plt.title("Newton error")
plt.grid()

plt.subplot(1,2,2)
plt.plot(nodes,errors_fact,"o-")
plt.title("Factorial error")
plt.grid()

plt.show()
