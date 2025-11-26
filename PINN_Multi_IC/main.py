
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

# Read in npz files 
def read_pf_data(time, shift_id, dLen):
    pf_filename = to_absolute_path(f"data_02468/shift_{int(shift_id):02d}/PF_{int(time):05d}.npz")
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
    invar["shift_id"] = np.full_like(invar["x"], shift_id/(5-1)) # Total no of initial condition=9


    outvar = {}
    outvar["c"] = np.expand_dims(Image.flatten(), axis=-1)  
    #outvar["eta"] = np.expand_dims(Image.flatten(), axis=-1) #gives error 
    
    return invar, outvar
def read_eta_data(time, shift_id, dLen):
    eta_filename = to_absolute_path(f"data_02468/shift_{int(shift_id):02d}/ETA_{int(time):05d}.npz")
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
    invar["shift_id"] = np.full_like(invar["x"], shift_id/(5-1))
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
    x, y, t, shift_id = Symbol("x"), Symbol("y"), Symbol("t"), Symbol("shift_id")
 
    # define geometry
    dLen = float(511)
    cen=dLen/2
    rec = Rectangle((0, 0), (dLen, dLen))
    cir =Circle(center=(cen, cen), radius=12.0)
    channel_length = (0.0, dLen)
    channel_width = (0.0, dLen)
    box_bounds = {x: channel_length, y: channel_width}

    

    # define networks and nodes
    if cfg.custom.parameterized:
        input_keys = [
            Key("x"),
            Key("y"),
            Key("t"),
            Key("shift_id"),
                                                                        ]
    else:
        input_keys = [Key("x"), Key("y"), Key("t")]

    eta_net = FullyConnectedArch(
        input_keys= input_keys,
        output_keys=[ Key("eta")],
        periodicity={"x": channel_length, "y": channel_width},
        layer_size=256,
    )
  
    c_net = FullyConnectedArch(
        input_keys=input_keys,
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
    invar=[Key("x"), Key("y"),Key("t"), Key("shift_id")],
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
    shift_range={shift_id:(0,1)}
    # add initial timesteps constraints
    batch_size = 2048
    train_shift_ids = np.arange(0,5,2)
    #batch_per_epoch = 500
    for shift_id in train_shift_ids:
        for i, t in enumerate(np.arange(1,300,4)):
            invar_c, outvar_c = read_pf_data(t, shift_id, dLen)
            lambda_weighting = {}
            lambda_weighting["c"] = np.full_like(invar_c["x"], 0.01)
            timestep = PointwiseConstraint.from_numpy(
                nodes,
                invar_c,
                outvar_c, 
                batch_size,
                lambda_weighting=lambda_weighting,
            )
            domain.add_constraint(timestep, f"BC_c_shift{shift_id:02d}_t{t:05d}")
    for shift_id in train_shift_ids:
        for i, t in enumerate(np.arange(1,300,4)):
            invar_eta, outvar_eta = read_eta_data(t, shift_id, dLen)
            lambda_weighting = {}
            lambda_weighting["eta"] = np.full_like(invar_eta["x"], 0.01)
            timestep = PointwiseConstraint.from_numpy(
                nodes,
                invar_eta,
                outvar_eta,
                batch_size,
                lambda_weighting=lambda_weighting,
            )
            domain.add_constraint(timestep, f"BC_eta_shift{shift_id:02d}_t{t:05d}")

    # add interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"CahnHilliard_equation":0,"AllenCahn_equation": 0},
        batch_size=2048,
        #batch_per_epoch = 500,
        bounds={x: (0, dLen), y: (0, dLen)},
        lambda_weighting={"CahnHilliard_equation": 0.00001,"AllenCahn_equation": 0.0001},
        parameterization={**time_range, **shift_range},
        importance_measure=dynamic_importance,
    )
    domain.add_constraint(interior, "Interior")


    for shift_id in train_shift_ids:
        for i, t in enumerate(np.arange(0, 1,1)):
            invar_c, outvar_c = read_pf_data(t, shift_id, dLen)
            invar_eta, outvar_eta = read_eta_data(t, shift_id, dLen)

            domain.add_constraint(PointwiseConstraint.from_numpy(nodes, invar_c, outvar_c, batch_size), f"IC_c_shift{shift_id:02d}_t{t:05d}" )
            domain.add_constraint(PointwiseConstraint.from_numpy(nodes, invar_eta, outvar_eta, batch_size), f"IC_eta_shift{shift_id:02d}_t{t:05d}")
        

    # add validators
    test_shift_ids = np.arange(1, 5, 2)
    for shift_id in test_shift_ids:
        for i, t in enumerate(np.arange(1, 500,50)):
            val_invar, val_true_outvar = read_pf_data(t, shift_id, dLen)
            validator = PointwiseValidator(                                                     nodes=nodes,                                                                    invar=val_invar,                                                                true_outvar=val_true_outvar,
                batch_size=2048,
                                                                                                )
            domain.add_validator(validator, f"VAL_shift{shift_id:02d}_t{t:05d}")
    for shift_id in test_shift_ids:
        for i, t in enumerate(np.arange(1, 500,50)):
            val_invar, val_true_outvar = read_eta_data(t, shift_id, dLen)
            validator = PointwiseValidator(
                nodes=nodes,
                invar=val_invar,
                true_outvar=val_true_outvar,
                batch_size=2048,
                )
            domain.add_validator(validator, f"VALeta_shift{shift_id:02d}_t{t:05d}")





    slv = Solver(cfg, domain)

    slv.solve()


if __name__ == "__main__":
    run()



