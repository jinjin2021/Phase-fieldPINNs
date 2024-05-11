
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from sympy import Symbol, Function, Number, Eq, Abs, sin, cos
import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.domain import Domain
from modulus.models.fully_connected import FullyConnectedArch
from modulus.key import Key
from modulus.node import Node
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from modulus.domain.validator import PointwiseValidator
from modulus.geometry.primitives_1d import Line1D,Point1D
from modulus.geometry.primitives_2d import Rectangle, Circle
from modulus.key import Key
from ACbinary_equation_1D import ACbinaryEquation
from Mu_equation_1D import MuEquation2D
from CahnHilliard_equation_1D import CahnHilliardEquation2D
from modulus.eq.pde import PDE
from modulus.utils.io.plotter import ValidatorPlotter


# Read in npz files generated using finite difference simulator Devito
def read_pf_data(time, Nx):
    pf_filename = to_absolute_path(f"1D_data/td500_t0.2/PF_{int(time):03d}.npz")
    #data = np.atleast_1d(pf_filename.arr_0)[0]
    Image = np.load(pf_filename)["arr_0"].astype(np.float32)
    #x=np.linspace(1,Nx,512)# 1 is out of range
    x = np.expand_dims(np.linspace(0, Nx, 512), axis=-1)
   # print("The shape of x is",x.shape) #(512,1)
    #print(type(Image))
    #invar={}
    #invar["x"] = np.expand_dims(np.linspace(0, Nx, 512), axis=-1)
    #print(type(invar["x"]))
    #invar["t"] = np.full_like(invar["x"], time * 0.005)
    #t=time*0.005
    #invar["t"]=np.expand_dims(t, axis=-1)
    t=np.full_like(x, time * (1/300))
    


    #print(t.shape)
    invar_numpy = {"x":x,"t": t}
    c=np.expand_dims(Image, axis=-1)
    #print("The shape of c is",c.shape)
    outvar_numpy = {"c": c}
    #invar["x"] = x
    #invar["t"] = np.full_like(invar["x"], time * 0.005)
    #print(invar["t"].shape) #(512,1)
    #outvar={}
    #outvar["c"]=np.expand_dims(Image, axis=-1)
    #print(outvar["c"].shape) #(512,1)
    return invar_numpy, outvar_numpy
    #return invar, outvar

def read_eta_data(time, Nx):
    eta_filename = to_absolute_path(f"1D_data/td500_t0.2/ETA_{int(time):03d}.npz")
    eta_Image = np.load(eta_filename)["arr_0"].astype(np.float32)
    x = np.expand_dims(np.linspace(0, Nx, 512), axis=-1)
    #invar={}
    #invar["x"]=x
    #t=time*0.005
    #invar["t"]=np.expand_dims(t, axis=-1)
    #invar["t"] = np.full_like(invar["x"], time * 0.005)
    #outvar = {}
    #outvar["eta"]=np.expand_dims(eta_Image, axis=-1)
    t=np.full_like(x, time * (1/300))
    #X, T = np.meshgrid(x, t)
    #X = np.expand_dims(X.flatten(), axis=-1)
    #T = np.expand_dims(T.flatten(), axis=-1)
    #eta, eta = np.meshgrid(eta_Image, eta_Image)
    #eta = np.expand_dims(eta.flatten(), axis=-1)
    invar_numpy = {"x":x,"t": t}
    eta=np.expand_dims(eta_Image, axis=-1)
    outvar_numpy = {"eta": eta}

    return invar_numpy, outvar_numpy
    #return invar, outvar


class WavePlotter(ValidatorPlotter):
   # "Define custom validator plotting class"

    def __call__(self, invar, true_outvar, pred_outvar):

        # only plot x,y dimensions
        invar = {k: v for k, v in invar.items() if k in ["x"]}
        fs = super().__call__(invar, true_outvar, pred_outvar)
        return fs

#Creating a Neural Network Node

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

      
    # define PDEs
    eq = MuEquation2D(eta="eta", c="c", Kc=1.0, dim=1, time=True)
    ch = CahnHilliardEquation2D(eta="eta", c="c", Mu=eq.equations["Mu"], M=1.0, dim=1, time=True)
    ac = ACbinaryEquation(eta="eta", c="c", Keta=1.0, L=1.0, dim=1, time=True)


    # define sympy domain variables
    x, t = Symbol("x"), Symbol("t")
 
    # define geometry
    Nx=float(511)
    geo = Line1D(0, Nx)
    channel_length = (0.0, Nx)
    box_bounds = {x: channel_length}

    

    # define networks and nodes
    eta_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[ Key("eta")],
        periodicity={"x": channel_length},
        layer_size=256,
    )
  
    c_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("c")],
        periodicity={"x": channel_length},
        layer_size=256,
    )

    nodes = (
        ch.make_nodes()
        + ac.make_nodes()
        + eq.make_nodes()
        + [ eta_net.make_node(name="eta_network")]
        + [c_net.make_node(name="c_network")]
        )
    #print(nodes.type): list
    # define time range
    time_length = 1
    time_range = {t: (0, time_length)}

    # make domainPF1D_data_512.py
    domain = Domain()
    # add initial constraint

       
    # add initial timesteps constraints
    batch_size = 128
    
    for i, ms in enumerate(np.arange(4,300,4)):
        timestep_invar_numpy, timestep_outvar_numpy = read_pf_data(ms, Nx)
        lambda_weighting = {}
        lambda_weighting["c"] = np.full_like(timestep_invar_numpy["x"], 0.01)
        timestep = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar_numpy,
            timestep_outvar_numpy,
            batch_size,
            lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(timestep, f"BC{i:03d}")
    
    for i, ms in enumerate(np.arange(4,300,4)):
        timestep_invar, timestep_outvar = read_eta_data(ms, Nx)
        lambda_weighting = {}
        lambda_weighting["eta"] = np.full_like(timestep_invar["x"], 0.01)
        timestep = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar,
            timestep_outvar,
            batch_size,
            lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(timestep, f"BC{i:04d}")

    
    # add interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"CahnHilliard_equation":0, "AllenCahn_equation": 0},
        batch_size=1024,
        bounds=box_bounds,
        lambda_weighting={"CahnHilliard_equation": 0.00001, "AllenCahn_equation": 0.0001},
        parameterization=time_range,
    )
    domain.add_constraint(interior, "Interior")
    
    for i, ms in enumerate(np.arange(0, 1,1)):
        timestep_invar_numpy, timestep_outvar_numpy = read_pf_data(ms, Nx)
        #lambda_weighting = {}
        #lambda_weighting["c"] = np.full_like(timestep_invar["x"], 10.0 / 1024)
        ic = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar_numpy,
            timestep_outvar_numpy,
            batch_size=128,
           # lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(ic,f"ic{i:04d}")
    for i, ms in enumerate(np.arange(0,1,1)):
        timestep_invar, timestep_outvar = read_eta_data(ms, Nx)
      #  lambda_weighting = {}
       # lambda_weighting["eta"] = np.full_like(timestep_invar["x"], 10.0 / 1024)
        ic = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar,
            timestep_outvar,
            batch_size=128,
        #    lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(ic, f"ic{i:04d}")

   
    # add validators
    for i, ms in enumerate(np.arange(1, 500,4)):
        val_invar, val_true_outvar = read_pf_data(ms, Nx)
        validator = PointwiseValidator(
            nodes=nodes,
            invar=val_invar,
            true_outvar=val_true_outvar,
            batch_size=1024,
            #plotter=WavePlotter(),
        )
        domain.add_validator(validator, f"VAL_{i:04d}")

    slv = Solver(cfg, domain)

    slv.solve()


if __name__ == "__main__":
    run()


