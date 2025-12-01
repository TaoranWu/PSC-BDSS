from __future__ import annotations
import gurobipy as gp
import numpy as np


class Solver_coe_tau:
    def __init__(self, name: str = "PAC_CBF", num_vars: int = 0, verbose: int = 0):
        self.solver_model = gp.Model(name)
        self.solver_model.setParam('OutputFlag', verbose)

        self.coe = self.solver_model.addMVar(shape=(num_vars + 2,), lb=0,
                                             vtype=gp.GRB.CONTINUOUS, name="coe")

    def set_init_value(self, init_value):
        self.coe.start = init_value

    def set_objective(self, h_x, minimize=False):
        obj = np.sum(h_x, axis=0) / h_x.shape[0]
        obj = obj.reshape(1, -1)
        obj = np.hstack((obj, np.array([[-5]])))
        obj = np.hstack((obj, np.array([[1]])))
        if minimize:
            self.solver_model.setObjective(obj @ self.coe, gp.GRB.MINIMIZE)
        else:
            self.solver_model.setObjective(obj @ self.coe, gp.GRB.MAXIMIZE)

    def clean_constraint(self):
        self.solver_model.remove(self.solver_model.getConstrs())
        self.solver_model.update()

    def add_constraint(self, cons, constant=0):
        cons_coe_bound = np.ones(cons.shape[1] - 2)
        cons_coe_bound = np.diag(cons_coe_bound)

        new_col_2 = np.zeros((cons_coe_bound.shape[0], 1))
        cons_coe_bound = np.hstack((cons_coe_bound, new_col_2))

        new_col = -1 * np.ones((cons_coe_bound.shape[0], 1))
        cons_coe_bound = np.hstack((cons_coe_bound, new_col))

        cons = np.append(cons, cons_coe_bound, axis=0)

        constant = np.concatenate((constant, np.zeros(cons.shape[1] - 2)))

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
