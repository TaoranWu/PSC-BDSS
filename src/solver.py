from __future__ import annotations
import gurobipy as gp
import numpy as np


class Solver:
    def __init__(self, name: str = "PAC_CBF", num_vars: int = 0, coe_lb=-1e3, coe_ub=1e3, verbose: int = 0):
        self.solver_model = gp.Model(name)
        self.solver_model.setParam('OutputFlag', verbose)

        self.coe = self.solver_model.addMVar(shape=(num_vars,), lb=coe_lb, ub=coe_ub,
                                             vtype=gp.GRB.CONTINUOUS, name="coe")

    def set_init_value(self, init_value):
        self.coe.start = init_value

    def set_objective(self, h_x, minimize=False):
        obj = np.sum(h_x, axis=0) / h_x.shape[0]
        obj = obj.reshape(1, -1)
        if minimize:
            self.solver_model.setObjective(obj @ self.coe, gp.GRB.MINIMIZE)
        else:
            self.solver_model.setObjective(obj @ self.coe, gp.GRB.MAXIMIZE)

    def clean_constraint(self):
        self.solver_model.remove(self.solver_model.getConstrs())
        self.solver_model.update()

    def add_constraint(self, cons, constant=0):
        self.solver_model.addConstr(cons @ self.coe <= constant, name="cons")

    def add_constraint_verification(self, cons_safe, cons_unsafe, C):
        self.solver_model.addConstr(cons_safe @ self.coe <= 0, "cons_safe")
        self.solver_model.addConstr(cons_unsafe @ self.coe <= C, "cons_unsafe")

    def solve(self):
        self.solver_model.update()
        self.solver_model.optimize()

        assert self.solver_model.status == gp.GRB.OPTIMAL
        print("Optimal objective: ", self.solver_model.objVal)
        print(f"Solving time of Gurobi: {self.solver_model.Runtime:.4f} seconds")

        return self.coe.X, self.solver_model.objVal

    def dispose(self):
        self.solver_model.dispose()
