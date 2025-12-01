from __future__ import annotations
import gurobipy as gp
import numpy as np


class SolverXi:
    def __init__(self, name: str = "PAC_CBF", num_vars: int = 0, coe_lb=None, coe_ub=None, xi_ub=None,
                 verbose: int = 1):
        self.solver_model = gp.Model(name)
        self.solver_model.setParam('OutputFlag', verbose)
        # self.solver_model.setParam('MIPGap', 1e-8)
        # self.solver_model.setParam('OptimalityTol', 1e-8)
        # self.solver_model.setParam('FeasibilityTol', 1e-6)
        self.num_vars = num_vars

        # lb = np.full(num_vars, coe_lb)
        # ub = np.full(num_vars, coe_ub)
        # lb[0] = 0.0
        # ub[0] = gp.GRB.INFINITY
        if coe_lb is not None and coe_ub is not None:
            self.var = self.solver_model.addMVar(shape=(num_vars,), lb=coe_lb, ub=coe_ub,
                                             vtype=gp.GRB.CONTINUOUS, name="var")
        elif coe_lb is not None and coe_ub is None:
            self.var = self.solver_model.addMVar(shape=(num_vars,), lb=coe_lb,
                                                 vtype=gp.GRB.CONTINUOUS, name="var")
        elif coe_lb is None and coe_ub is not None:
            self.var = self.solver_model.addMVar(shape=(num_vars,), ub=coe_ub,
                                                 vtype=gp.GRB.CONTINUOUS, name="var")
        else:
            self.var = self.solver_model.addMVar(shape=(num_vars,), vtype=gp.GRB.CONTINUOUS, name="var")

    def set_init_value(self, init_value):
        self.var.start = init_value

    def clean_constraint(self):
        self.solver_model.remove(self.solver_model.getConstrs())
        self.solver_model.update()

    def add_constraint(self, cons, constant=0):
        # cons_xi = np.zeros(self.num_vars)
        # cons_xi[0] = -1
        # cons_xi = cons_xi.reshape(1, self.num_vars)
        # cons = np.append(cons, cons_xi, axis=0)
        # if isinstance(constant, np.ndarray):
        #     constant = np.append(constant, 0)

        self.solver_model.addConstr(cons @ self.var <= constant, name="cons")

    def solve(self):
        obj = np.zeros(self.num_vars)
        obj[0] = 1
        self.solver_model.setObjective(obj @ self.var, gp.GRB.MINIMIZE)

        self.solver_model.update()
        self.solver_model.optimize()

        assert self.solver_model.status == gp.GRB.OPTIMAL
        print(f"Optimal objective: {self.solver_model.objVal:.8f}")

        print(f"Solving time of Gurobi: {self.solver_model.Runtime:.4f} seconds")

        return self.var.X
