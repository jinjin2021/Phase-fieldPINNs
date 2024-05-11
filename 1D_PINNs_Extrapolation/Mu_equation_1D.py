from sympy import Symbol, Function, Number
from modulus.eq.pde import PDE
class MuEquation2D(PDE):
    name = "MuEquation2D"
    def __init__(self, eta="eta", c="c", Kc=1.0, dim=1, time=True):
        #set params
        self.eta=eta
        self.c=c
        self.Kc=Kc
        self.dim=dim
        self.time=time
        # coordinates
        x = Symbol("x")
        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "t": t}

        c = Function("c")(*input_variables)
        eta = Function("eta")(*input_variables)

        #if type(fc) is str:
           # fc = Function(fc)(*input_variables)
        #elif type(fc) in [float, int]:
           # fc = Number(fc)

        if type(Kc) is str:
            Kc = Function(Kc)(*input_variables)
        elif type(Kc) in [float, int]:
            Kc = Number(Kc)
       # if type(cAlpha_eq) is str:
           #cAlpha_eq = Function(CAlpha_eq)(*input_variables)
        #elif type(cAlpha_eq) in [float, int]:
           # cAlpha_eq = Number(cAlpha_eq)
        #if type(cBeta_eq) is str:
           # cBeta_eq = Function(cBeta_eq)(*input_variables)
        #elif type(cBeta_eq) in [float, int]:
            #cBeta_eq = Number(cBeta_eq)
        #h=10 * (eta **3)- 15 * (eta **4) + 6 *(eta**5)
        #fc=2 * (1-h)* (c-cAlpha_eq)- 2 * h * (cBeta_eq-c)
        interp_eta = eta * eta * (3.0 - 2.0 * eta)
        fc= 2.0  * c * (1.0 - interp_eta) - 2.0  * (1.0 - c) * interp_eta
        # set equations
        self.equations = {}
        self.equations["Mu"] = fc-2 * Kc* (c.diff(x,2))
