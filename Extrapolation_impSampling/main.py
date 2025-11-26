import time
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
from modulus.graph import Graph
from modulus.domain.constraint import Constraint

# Read in npz files generated using finite difference simulator Devito
def read_pf_data(time, dLen):
    pf_filename = to_absolute_path(f"PINNs2D_data/imp_50pTD512t0.2_seed100/PF_{int(time):05d}.npz")
    Image = np.load(pf_filename)["arr_0"].astype(np.float32)
    mesh_y, mesh_x = np.meshgrid(
        np.linspace(0, dLen, Image.shape[0]),
        np.linspace(0, dLen, Image.shape[1]),
        indexing="ij",
    )
    invar = {}
    invar["x"] = np.expand_dims(mesh_y.astype(np.float32).flatten(), axis=-1)
    invar["y"] = np.expand_dims(mesh_x.astype(np.float32).flatten(), axis=-1)
    invar["t"] = np.full_like(invar["x"], time * 1/500)
    outvar = {}
    outvar["c"] = np.expand_dims(Image.flatten(), axis=-1)  
    #outvar["eta"] = np.expand_dims(Image.flatten(), axis=-1) #gives error 
    
    return invar, outvar
def read_eta_data(time, dLen):
    eta_filename = to_absolute_path(f"PINNs2D_data/imp_50pTD512t0.2_seed100/ETA_{int(time):05d}.npz")
    eta_Image = np.load(eta_filename)["arr_0"].astype(np.float32)
    mesh_y, mesh_x = np.meshgrid(
        np.linspace(0, dLen, eta_Image.shape[0]),
        np.linspace(0, dLen, eta_Image.shape[1]),
        indexing="ij",
    )
    invar = {}
    invar["x"] = np.expand_dims(mesh_y.astype(np.float32).flatten(), axis=-1)
    invar["y"] = np.expand_dims(mesh_x.astype(np.float32).flatten(), axis=-1)
    invar["t"] = np.full_like(invar["x"], time * 1/500)# 500*0.002=1
    outvar = {}
    outvar["eta"] = np.expand_dims(eta_Image.flatten(), axis=-1)  
   
    
    return invar, outvar



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
    dLen = float(511)
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

# make importance model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    importance_model_graph = Graph(
    nodes,
    invar=[Key("x"), Key("y"),Key("t")],
    req_names=[Key("eta")],
    ).to(device)

    def importance_measure(invar):
        outvar = importance_model_graph(
        Constraint._set_device(invar, device=device, requires_grad=True)
        )
        eta = outvar["eta"]
        importance = ((eta > 0) & (eta < 1)).float() + 0.01  # add a small c    onstant to     avoid zero weight
        return importance.cpu().detach().numpy()
 
    class DynamicImportance:
        def __init__(self, importance_fn, start_epoch=30000):
            self.importance_fn = importance_fn
            self.start_epoch = start_epoch
            self.current_epoch = 0
 
        def set_epoch(self, epoch):
            self.current_epoch = epoch
 
        def __call__(self, invar):
            if self.current_epoch < self.start_epoch:
             # Return uniform importance
                return np.ones_like(invar["x"])
            else:
                return self.importance_fn(invar)
    dynamic_importance = DynamicImportance(importance_measure, start_epoch=30000)




    # define time range
    time_length = 1.0
    time_range = {t: (0.0, time_length)}

    # make domain
    domain = Domain()

    # add velocity constraint
   
    # add initial timesteps constraints
    batch_size = 4096
    #batch_per_epoch = 500
    for i, ms in enumerate(np.arange(1,500,4)):
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
    
    for i, ms in enumerate(np.arange(1,500,4)):
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
        batch_size=4096,
        bounds={x: (0, dLen), y: (0, dLen)},
        lambda_weighting={"CahnHilliard_equation": 0.00001,"AllenCahn_equation": 0.0001},
        parameterization=time_range,
        importance_measure=dynamic_importance,
    )
    domain.add_constraint(interior, "Interior")
    for i, ms in enumerate(np.arange(0, 1,1)):
        timestep_invar_numpy, timestep_outvar_numpy = read_pf_data(ms, dLen)
        ic = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar_numpy,
            timestep_outvar_numpy,
            batch_size,
  
        )
        domain.add_constraint(ic,f"ic{i:04d}")
    for i, ms in enumerate(np.arange(0,1,1)):
        timestep_invar, timestep_outvar = read_eta_data(ms, dLen)

        ic = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar,
            timestep_outvar,
            batch_size,
        )
        domain.add_constraint(ic, f"ic{i:04d}")



    # add validators
    for i, ms in enumerate(np.arange(1, 500,100)):
        val_invar, val_true_outvar = read_pf_data(ms, dLen)
        validator = PointwiseValidator(
            nodes=nodes,
            invar=val_invar,
            true_outvar=val_true_outvar,
            batch_size=4096,
        )
        domain.add_validator(validator, f"VAL_{ms:05d}")

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()



