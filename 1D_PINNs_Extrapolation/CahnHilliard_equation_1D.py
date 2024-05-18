from sympy import Symbol, Function, Number
from modulus.eq.pde import PDE
class CahnHilliardEquation2D(PDE):
    name = "CahnHilliardEquation2D"
    def __init__(self, eta="eta", c="c", Mu= "Mu", M=1.0, dim=1, time=True):
        #set params
        self.eta=eta
        self.c=c
        self.Mu=Mu
        self.M=M
        self.dim=dim
        self.time=time

        # coordinates
        x = Symbol("x")
        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "t": t}

        # Define the function for eta,.., that are dependent on the input variabales(x,y,t)
        eta = Function("eta")(*input_variables)
        c=Function("c")(*input_variables)
        Mu=Function("Mu")(*input_variables)
       
        # Kappa(Kc) coefficient
        #Converting a string to functional form allow us to solve problems with  spatially/temporally varying properties.
        #if type(cAlpha_eq) is str:
          # cAlpha_eq = Function(CAlpha_eq)(*input_variables)
        #elif type(cAlpha_eq) in [float, int]:
           # cAlpha_eq = Number(cAlpha_eq)
        #if type(cBeta_eq) is str:
           # cBeta_eq = Function(cBeta_eq)(*input_variables)
        #elif type(cBeta_eq) in [float, int]:
           # cBeta_eq = Number(cBeta_eq)
        if type(M) is str:
            M = Function(M)(*input_variables)
        elif type(M) in [float, int]:
            M = Number(M)
        #if type(Mu) is str:
           # Mu = Function(Mu)(*input_variables)
        #elif type(Mu) in [float, int]:
           # Mu = Number(Mu)
       
       
        #h=10 * (eta **3)- 15 * (eta **4) + 6 *(eta**5)
        #fc=2 * (1-h)* (c-cAlpha_eq)- 2 * h * (cBeta_eq-c)
        #kc=1
        # set equations
        self.equations = {}
        self.equations["CahnHilliard_equation"]=c.diff(t)- M * (Mu.diff(x,2))     



