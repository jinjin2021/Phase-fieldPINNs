
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
from modulus.geometry.primitives_2d import Rectangle, Circle
from modulus.key import Key
from ACbinary_equation import ACbinaryEquation
from Mu_equation import MuEquation2D
from CahnHilliard_equation import CahnHilliardEquation2D
from modulus.eq.pde import PDE
from modulus.utils.io.plotter import ValidatorPlotter


# Read in npz files generated using finite difference simulator Devito
def read_pf_data(time, dLen):
    pf_filename = to_absolute_path(f"Mprecipitate_n1/50pTD512t0.2_seed100/PF_{int(time):05d}.npz")
    Image = np.load(pf_filename)["arr_0"].astype(np.float32)
    mesh_y, mesh_x = np.meshgrid(
        np.linspace(0, dLen, Image.shape[0]),
        np.linspace(0, dLen, Image.shape[1]),
        indexing="ij",
    )
    invar = {}
    invar["x"] = np.expand_dims(mesh_y.astype(np.float32).flatten(), axis=-1)
    invar["y"] = np.expand_dims(mesh_x.astype(np.float32).flatten(), axis=-1)
    invar["t"] = np.full_like(invar["x"], time * 1/300)
    outvar = {}
    outvar["c"] = np.expand_dims(Image.flatten(), axis=-1)  
    #outvar["eta"] = np.expand_dims(Image.flatten(), axis=-1) #gives error 
    
    return invar, outvar
def read_eta_data(time, dLen):
    eta_filename = to_absolute_path(f"Mprecipitate_n1/50pTD512t0.2_seed100/ETA_{int(time):05d}.npz")
    eta_Image = np.load(eta_filename)["arr_0"].astype(np.float32)
    mesh_y, mesh_x = np.meshgrid(
        np.linspace(0, dLen, eta_Image.shape[0]),
        np.linspace(0, dLen, eta_Image.shape[1]),
        indexing="ij",
    )
    invar = {}
    invar["x"] = np.expand_dims(mesh_y.astype(np.float32).flatten(), axis=-1)
    invar["y"] = np.expand_dims(mesh_x.astype(np.float32).flatten(), axis=-1)
    invar["t"] = np.full_like(invar["x"], time * 1/300)# 500*0.002=1
    outvar = {}
    outvar["eta"] = np.expand_dims(eta_Image.flatten(), axis=-1)  
   
    
    return invar, outvar


#



class WavePlotter(ValidatorPlotter):
    "Define custom validator plotting class"

    def __call__(self, invar, true_outvar, pred_outvar):

        # only plot x,y dimensions
        invar = {k: v for k, v in invar.items() if k in ["x", "y"]}
        fs = super().__call__(invar, true_outvar, pred_outvar)
        return fs

#Creating a Neural Network Node

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

      
    # define PDEs
    eq = MuEquation2D(eta="eta", c="c", cAlpha_eq=0.0, cBeta_eq=1.0, Kc=1.0, dim=2, time=True)
    ch = CahnHilliardEquation2D(eta="eta", c="c", Mu=eq.equations["Mu"], M=1.0, dim=2, time=True)
    ac = ACbinaryEquation(eta="eta", c="c", cAlpha_eq=0.0, cBeta_eq=1.0, omega=1.0, Keta=1.0, L=1.0, dim=2, time=True)


    # define sympy domain variables
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
 
    # define geometry
    dLen = 512.0  # km
    cen=dLen/2
    rec = Rectangle((0, 0), (dLen, dLen))
    cir =Circle(center=(cen, cen), radius=12.0)
    channel_length = (0.0, dLen)
    channel_width = (0.0, dLen)
    box_bounds = {x: channel_length, y: channel_width}

    

    # define networks and nodes
    eta_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[ Key("eta")],
        periodicity={"x": channel_length, "y": channel_width},
        layer_size=256,
    )
  
    c_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("c")],
        periodicity={"x": channel_length, "y": channel_width},
        layer_size=256,
    )

    nodes = (
        ch.make_nodes()
        + ac.make_nodes()
        + eq.make_nodes()
        + [ eta_net.make_node(name="eta_network")]
        + [c_net.make_node(name="c_network")]
        ) 
    # define time range
    time_length = 1.0
    time_range = {t: (0.0, time_length)}

    # define target velocity model
    # 2.0 km/s at the bottom and 1.0 km/s at the top using tanh function

    # make domain
    domain = Domain()

    # add velocity constraint
   
    # add initial timesteps constraints
    batch_size = 5000
    #batch_per_epoch = 500
    for i, ms in enumerate(np.arange(1,51,1)):
        timestep_invar, timestep_outvar = read_pf_data(ms, dLen)
        lambda_weighting = {}
        lambda_weighting["c"] = np.full_like(timestep_invar["x"], 0.01)
        timestep = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar,
            timestep_outvar,
            batch_size,
            lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(timestep, f"BC{i:05d}")
    
    for i, ms in enumerate(np.arange(1,51,1)):
        timestep_invar, timestep_outvar = read_eta_data(ms, dLen)
        lambda_weighting = {}
        lambda_weighting["eta"] = np.full_like(timestep_invar["x"], 0.01)
        timestep = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar,
            timestep_outvar,
            batch_size,
            lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(timestep, f"BC{i:05d}")

    # add interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"CahnHilliard_equation":0,"AllenCahn_equation": 0},
        batch_size=5000,
        #batch_per_epoch = 500,
        bounds={x: (0, dLen), y: (0, dLen)},
        lambda_weighting={"CahnHilliard_equation": 0.00001,"AllenCahn_equation": 0.0001},
        parameterization=time_range,
    )
    domain.add_constraint(interior, "Interior")
    for i, ms in enumerate(np.arange(0, 1,1)):
        timestep_invar_numpy, timestep_outvar_numpy = read_pf_data(ms, dLen)
        #lambda_weighting = {}
        #lambda_weighting["c"] = np.full_like(timestep_invar["x"], 10.0 / 1024)
        ic = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar_numpy,
            timestep_outvar_numpy,
            batch_size,
           # lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(ic,f"ic{i:04d}")
    for i, ms in enumerate(np.arange(0,1,1)):
        timestep_invar, timestep_outvar = read_eta_data(ms, dLen)
      #  lambda_weighting = {}
       # lambda_weighting["eta"] = np.full_like(timestep_invar["x"], 10.0 / 1024)
        ic = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar,
            timestep_outvar,
            batch_size,
        #    lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(ic, f"ic{i:04d}")

# c0=0.05, cBeta_eq=1.0
    # add open boundary constraint
   # ic = PointwiseInteriorConstraint(
       # nodes=nodes,
       # geometry=rec-cir,
       # outvar={"c": 0.05, "eta": 0.0},
       # batch_size=1024,
       # bounds=box_bounds,
       # lambda_weighting={"c": 0.01 * time_length, "eta": 0.01 * time_length},
       # parameterization=time_range,
    #)
    #domain.add_constraint(ic, "ic")
    
    #ic_eta = PointwiseInteriorConstraint(
       # nodes=nodes,
        #geometry=rec-cir,
       # outvar={"eta": 0.0},
       # batch_size=1024,
       # bounds=box_bounds,
       # lambda_weighting={"eta": 0.01 * time_length},
       # parameterization=time_range,
    #)
    #domain.add_constraint(ic_eta, "ic_eta")

    #ic_cir = PointwiseInteriorConstraint(
     #   nodes=nodes,
      #  geometry=cir,
       # outvar={"c": 1.0, "eta": 1.0},
       # batch_size=1024,
       # bounds=box_bounds,
       # lambda_weighting={"c": 0.01 * time_length, "eta": 0.01 * time_length},
       # parameterization=time_range,
    #)
    #domain.add_constraint(ic_cir, "ic_cir")

    #ic_cir_eta = PointwiseInteriorConstraint(
       # nodes=nodes,
       # geometry=cir,
       # outvar={"eta": 1.0},
       # batch_size=1024,
       # bounds=box_bounds,
       # lambda_weighting={"eta": 0.01 * time_length},
       # parameterization=time_range,
    #)
    #domain.add_constraint(ic_cir_eta, "ic_cir_eta")

    # add validators
    for i, ms in enumerate(np.arange(1, 500,4)):
        val_invar, val_true_outvar = read_pf_data(ms, dLen)
        validator = PointwiseValidator(
            nodes=nodes,
            invar=val_invar,
            true_outvar=val_true_outvar,
            batch_size=5000,
           # plotter=WavePlotter(),
        )
        domain.add_validator(validator, f"VAL_{i:05d}")

    slv = Solver(cfg, domain)

    slv.solve()


if __name__ == "__main__":
    run()



