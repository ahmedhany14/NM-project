import numpy as np
import pandas as pd

"""
    3
    400
    100
    210
    200
    410
    300
    610
    
    3
    100 210
    200 410
    300 610
    
    400
    810

"""


def bisection_method(a, b, Function, error, max_iterations=1000):

    def func(x, fucntion_of_x):
        return eval(fucntion_of_x, {"x": x})

    if func(a, Function) * func(b, Function) >= 0:
        return None, None

    c, it, condition = (a + b) / 2, 0, True
    columns = {
        "Iteration": [],
        "a": [],
        "b": [],
        "c": [],
        "f(a)": [],
        "f(b)": [],
        "f(c)": [],
        "error": [],
    }
    table_of_iterations = pd.DataFrame(columns=columns)
    while (condition or func(c, Function) >= error) and it < max_iterations:
        c = (a + b) / 2
        it += 1
        if func(c, Function) == 0.0:
            break
        if func(a, Function) * func(b, Function) >= 0:
            return None, None
        if func(c, Function) * func(a, Function) < 0:
            b = c
        else:
            a = c
        condition = abs(b - a) >= error
        data = {
            "Iteration": [it],
            "a": a,
            "b": b,
            "c": c,
            "f(a)": func(a, Function),
            "f(b)": func(b, Function),
            "f(c)": func(c, Function),
            "error": abs(b - a),
        }
        table_of_iterations.loc[len(table_of_iterations)] = data
    return c, table_of_iterations


"""
class NumericalMethods:

    def FX(self, x, equ):
        return eval(equ)

    def Bisection(self):
        eq = input("enter an equation in Python form \n")
        l = float(input("enter the left border\n"))
        r = float(input("enter the right border\n"))
        E = float(input("enter the Error tolerance\n"))
        while True:
            mid = (l + r) / 2
            fl = self.FX(l, eq)
            fr = self.FX(r, eq)
            fmid = self.FX(mid, eq)
            print(
                "l = {}\t, r = {}\t, mid = {}\t, f(l) = {}\t, f(r) = {}\t, f(mid) = {}\t".format(
                    l, r, mid, fl, fr, fmid
                )
            )
            if (fl > 0) != (fr > 0) and fl:
                if fmid > 0:
                    r = mid
                elif fmid < 0:
                    l = mid
                else:  # (x**3) + 4*(x**2) - 10
                    print("x = {}".format(mid))
                    break
            else:
                print("there is no solution in this constrains\n")
                break

    def interpolation(self):
        print("enter the first point")
        x0 = float(input("x0 = "))
        y0 = float(input("y0 = "))

        print("enter the second point")
        x1 = float(input("x1 = "))
        y1 = float(input("y1 = "))

        print("enter the test value")
        test = float(input("test = "))

        # a0 + a1X1 = fx1
        # a0 + a1X2 = fx2

        eq = np.array([[1, x0], [1, x1]])
        fx = np.array([y0, y1])

        a0, a1 = np.linalg.solve(eq, fx)
        print(test * a1 + a0)

    def lagrange(self):
        n = int(input("enter the number of points\n"))
        test = float(input("enter the test\n"))
        x = np.array([0 for i in range(n)])
        f = np.array([0 for i in range(n)])
        l = np.array([0 for i in range(n)])
        for i in range(n):
            print("Enter the {}th point".format(i))
            x[i] = float(input("x{}=  ".format(chr(i + 48))))
            f[i] = float(input("y{}=  ".format(chr(i + 48))))
        for i in range(n):
            pst = 1
            mqam = 1
            for j in range(n):
                if i == j:
                    continue
                else:
                    pst *= test - x[j]
            for j in range(n):
                if i == j:
                    continue
                else:
                    mqam *= x[i] - x[j]
            l[i] = pst / mqam
        sum = 0.0
        for i in range(n):
            sum += l[i] * f[i]
        print(sum)

    def NewtonDividedDiffirence(self):
        n = int(input("enter the number of points\n"))
        test = float(input("enter the test\n"))
        x = np.array([0 for i in range(n)], float)
        f = np.array([0 for i in range(n)], float)
        F = np.array([0 for i in range(n)], float)
        P = np.array([0 for i in range(n)], float)
        v = [[None for i in range(n)] for i in range(n)]
        for i in range(n):
            print("Enter the {}th point".format(i))
            x[i] = float(input("x{}=  ".format(chr(i + 48))))
            f[i] = float(input("y{}=  ".format(chr(i + 48))))

        v[0] = x.copy()
        v[1] = f.copy()

        start = 1
        dif = 1
        for i in range(2, n):
            l = 0
            r = dif

            for j in range(start, n):
                pst = v[i - 1][j] - v[i - 1][j - 1]
                mqam = x[r] - x[l]
                v[i][j] = float(pst / mqam)
                l += 1
                r += 1

            start += 1
            dif += 1
        for i in range(1, n):
            # print(*v[i])
            for it in v[i]:
                if it != None:
                    F[i - 1] = it
                    break
        P[0] = F[0]
        x[0] = test - x[0]
        for i in range(1, n + 1):
            x[i] = (test - x[i]) * x[i - 1]
            P[i] = F[i] * (x[i - 1]) + P[i - 1]

        for it in v:
            print(it, "\n")

        print(P[-1])

    def LeastSquares(self):
        n = int(input("enter the number of points\n"))
        test = float(input("enter the test\n"))
        x = np.array([0 for i in range(n)])
        y = np.array([0 for i in range(n)])
        for i in range(n):
            print("Enter the {}th point".format(i))
            x[i] = float(input("x{}=  ".format(chr(i + 48))))
            y[i] = float(input("y{}=  ".format(chr(i + 48))))

        X = sum(x)
        Y = sum(y)
        XY = sum(x * y)
        X2Y = sum(x * x * y)
        X2 = sum(x * x)
        X3 = sum(x * x * x)
        X4 = sum(x * x * x * x)

        print(X, Y, XY)
        co = np.array([[n, X, X2], [X, X2, X3], [X2, X3, X4]])
        val = np.array([Y, XY, X2Y])
        a0, a1, a2 = np.linalg.solve(co, val)
        print(a0, a1, a2)

    def NewtonsMethod(self, fun, funD, x, iterations):

        def f(x):
            return eval(fun)

        def df(x):
            dif = eval(funD)
            return dif

        for i in range(iterations):
            if df(x) == 0:
                print("the root of the function i x = {}")
            i = x - (f(x) / df(x))
            x = i
        print("the root of the function i x = {}".format(x))


obj = NumericalMethods()
obj.interpolation()
"""
