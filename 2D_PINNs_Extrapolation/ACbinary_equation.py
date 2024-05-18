from sympy import Symbol, Function, Number
from modulus.eq.pde import PDE
class ACbinaryEquation(PDE):
    name = "ACbinaryEquation"
    def __init__(self, eta="eta", c="c", cAlpha_eq=0.0, cBeta_eq=1.0, omega= 1.0, Keta=1.0, L=1.0, dim=2, time=True):
        # set params
        self.eta= eta
        self.c= c
        self.cAlpha_eq= cAlpha_eq
        self.cBeta_eq= cBeta_eq
        self.omega= omega
        self.Keta= Keta
        self.L= L
        self.dim= dim
        self.time= time
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
        if type(Keta) is str:
            Keta = Function(Keta)(*input_variables)
        elif type(Keta) in [float, int]:
            Keta = Number(Keta)

        # Mobility(M)
        if type(L) is str:
            L = Function(L)(*input_variables)
        elif type(L) in [float, int]:
            L = Number(L)
        if type(cAlpha_eq) is str:
           cAlpha_eq = Function(CAlpha_eq)(*input_variables)
        elif type(cAlpha_eq) in [float, int]:
            cAlpha_eq = Number(cAlpha_eq)
        if type(cBeta_eq) is str:
            cBeta_eq = Function(cBeta_eq)(*input_variables)
        elif type(cBeta_eq) in [float, int]:
            cBeta_eq = Number(cBeta_eq)
        if type(omega) is str:
            omega = Function(omega)(*input_variables)
        elif type(omega) in [float, int]:
            omega = Number(omega)
        falpha=(c-cAlpha_eq)**2
        fbeta=(cBeta_eq-c)**2
        hprime=30 * (eta**2) * (eta -1)**2
        gprime=2 * (eta-eta**2)* (1- 2 * eta)
        feta= hprime * (fbeta-falpha) + omega * gprime
        # set equations
        self.equations = {}
        self.equations["AllenCahn_equation"] = eta.diff(t) + L *feta- L*2*Keta*(eta.diff(x,2)+eta.diff(y,2))
    
