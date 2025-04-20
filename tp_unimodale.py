import math 

def dichotomie(f, a, b):
 
    while (b - a) > 1.e-7:
        x_g = a + 1/4 * (b - a)
        x_c = a + 1/2 * (b - a)
        x_d = a + 3/4 * (b - a)
        if f(x_c) > f(x_d):
            a = x_c
            b = b
        elif f(x_c) > f(x_g):
            a = a
            b = x_c
        else:
            a = x_g
            b = x_d
    return a,b



def f1(x):
    return abs(x-100)

def f2(x):
    if(x <= 50):
        return math.sqrt(-(x-50))
    else:
        return math.sqrt(x-50)

def f3(x):
    return min(4*x, x+5)

def f4(x):
    return -(x**3)



print(dichotomie(f1, -1000, 1000))
print(dichotomie(f2, -1000, 1000))
print(dichotomie(f3, -1000, 1000))
print(dichotomie(f4, -1000, 1000))



def dichotomie_2(f, a, b):
 
    while (b - a) > 1.e-7:
        x_g = a + 1/4 * (b - a)
        x_d = a + 3/4 * (b - a)
        if f(x_g) > f(x_d):
            a = x_g
            b = b
        elif f(x_g) < f(x_d):
            a = a
            b = x_d
        else:
            a = x_g
            b = x_d
    return a,b


def dichotomie_or(f, a, b):
    alpha = (1 + math.sqrt(5))/2
    xD = a + (b-a)/alpha
    xG = b - (b-a)/alpha
    f1 = f(xG)
    f2 = f(xD)
    while (b - a) > 1.e-7:
        if f1 > f2:
            a = xG
            xG = xD
            xD = a + (b-a)/alpha
            f1 = f2
            f2 = f(xD)
        elif f1 < f2:
            b = xD
            xD = xG
            xG = b - (b-a)/alpha
            f2 = f1
            f1 = f(xG)
        else:
            a = xG
            b = xD
            xD = a + (b-a)/alpha
            xG = b - (b-a)/alpha
            f1 = f(xG)
            f2 = f(xD)
    return a,b


print("Maintenant avec la méthode du nombre d'or")
print(dichotomie_or(f1, -1000, 1000))
print(dichotomie_or(f2, -1000, 1000))
print(dichotomie_or(f3, -1000, 1000))
print(dichotomie_or(f4, -1000, 1000))
