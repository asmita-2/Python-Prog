### This code shows the 


##Let x1 = [3, 6], x2 = [10, 10].  Use the above “Transform” function to transform these vectors to a higher dimension and
## compute the dot product in a higher dimension. Print the value.    Implement a polynomial kernel K(a,b) =
## a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2 . Apply this kernel function and evaluate the output for the
## same x1 and x2 values. Notice that the result is the same in both scenarios demonstrating the power of kernel trick.

def kernel(x1, x2):
    #k_func = a[0] ** 2 * b[0] ** 2 + 2 * a[0] * b[0] * a[1] * b[1] + a[1] ** 2 * b[1] ** 2
    k_trick = x1[0] ** 2 * x2[0] ** 2 + 2 * x1[0] * x2[0] * x1[1] * x2[1] + x1[1] ** 2 * x2[1] ** 2
    print("dot product using kernel", k_trick)
def dot_prod(X1, X2, x1, x2):
    result = np.dot(X1, X2)
    print("dot product in higher dim", result)
    kernel(x1, x2)
def transform(X1, X2, x1, x2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2)
    #plt.show()
    dot_prod(X1, X2, x1, x2)
def threeD(x1,x2):
    X1 = []
    X2 = []
    mul = 1
    mul2 = 1
    for i in range(len(x1)):
        mul = mul * x1[i]
    prod1 = mul
    for i in range(len(x1)):
        cal1a = x1[i]**2
        X1.append(cal1a)
    cal1b = prod1 * (2 ** 0.5)
    X1.append(cal1b)
    print("x1 after higher dim cal", X1)
    for i in range(len(x2)):
        mul2 = mul2 * x2[i]
    prod2 = mul2
    for i in range(len(x2)):
        cal1a = x2[i]**2
        X2.append(cal1a)
    cal1b = prod2 * (2 ** 0.5)
    X2.append(cal1b)
    print("x2 after higher dim cal", X2)
    transform(X1, X2, x1, x2)

def phi(x1,x2, label):
    plt.scatter(x1, x2, c=label)
    # plt.show()
    threeD(x1, x2)

def main():
    x1 = [3, 6]
    x2 = [10, 10]
    label = ['blue', 'red']
    phi(x1, x2, label)
if __name__=="__main__":
    main()

