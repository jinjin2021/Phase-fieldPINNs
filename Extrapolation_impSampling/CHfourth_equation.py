from sympy import Symbol, Function, Number
from modulus.eq.pde import PDE
class CHfourthEquation2D(PDE):
    name = "CHfourthEquation2D"
    def __init__(self, eta="eta", c="c", cAlpha_eq=0.0, cBeta_eq=1.0, M=1.0, dim=2, time=True):
        # coordinates
        x, y = Symbol("x"),Symbol("y")
        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "t": t}

        # Define the function for eta,.., that are dependent on the input variabales(x,y,t)
        eta = Function("eta")(*input_variables)
        c=Function("c")(*input_variables)
       
        # Kappa(Kc) coefficient
        #Converting a string to functional form allow us to solve problems with  spatially/temporally varying properties.
        if type(cAlpha_eq) is str:
            cAlpha_eq = Function(CAlpha_eq)(*input_variables)
        elif type(cAlpha_eq) in [float, int]:
            cAlpha_eq = Number(cAlpha_eq)
        if type(cBeta_eq) is str:
            cBeta_eq = Function(cBeta_eq)(*input_variables)
        elif type(cBeta_eq) in [float, int]:
            cBeta_eq = Number(cBeta_eq)
        if type(M) is str:
            M = Function(M)(*input_variables)
        elif type(M) in [float, int]:
            M = Number(M)
              
       
        h=10 * (eta **3)- 15 * (eta **4) + 6 *(eta**5)
        fc=2 * (1-h)* (c-cAlpha_eq)- 2 * h * (cBeta_eq-c)
        kc=1
        # set equations
        self.equations = {}
        self.equations["CHfourth_equation"]=c.diff(t)- M * (fc.diff(x,2)+fc.diff(y,2)) + 2* kc * M * ( c.diff(x,4)+ c.diff(y,4))     



