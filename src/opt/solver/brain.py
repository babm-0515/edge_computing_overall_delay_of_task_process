import time
import numpy as np
from typing import List
from pyomo.environ import value

from src.scenario.operate import Operate
from src.scenario.env import Env
from src.opt.solver.model.pym import PyoModel


class Brain:
    def __init__(self, env: Env, is_scheduled: bool, has_caching: bool, linear: bool = True, deletion: bool = False,
                 solver: str = 'scip'):
        self.opt = PyoModel(env=env, solver=solver)

        self.is_scheduled = is_scheduled
        self.has_caching = has_caching

        # deletion
        self.deletion = deletion

        if is_scheduled:
            self.opt.add_schedule_var_cons()
        if has_caching:
            self.opt.add_cache_var_cons()

        if is_scheduled and has_caching:
            if linear:
                if deletion:
                    # config deletion obj
                    self.opt.set_linear_plus_deletion_object()
                else:
                    self.opt.set_linear_object()
            else:
                self.opt.set_nonlinear_object()
        elif is_scheduled:
            self.opt.set_schedule_object()
        elif has_caching:
            self.opt.set_cache_object()
        else:
            self.opt.set_offload_object()

    def set_mutable_param(self, operate: Operate):
        t = operate.env.time_count
        for iot_index in range(operate.env.n_iot):
            self.opt.model.G[iot_index] = operate.queue_iot_comp_remain[iot_index]

            for edge_index in range(operate.env.n_edge):
                self.opt.model.U[iot_index, edge_index] = operate.queue_iot_tran_remain[iot_index, edge_index]
                self.opt.model.Q[iot_index, edge_index] = operate.queue_fog_comp_remain[iot_index, edge_index]

        for iot_index in range(len(self.opt.model.I)):
            self.opt.model.D[iot_index] = operate.env.bit_arrive[t][iot_index]

        for iot_index in range(operate.env.n_iot):
            for edge_index in range(operate.env.n_edge):
                self.opt.model.y_pre[iot_index, edge_index] = operate.y_pre[iot_index, edge_index]
                self.opt.model.z_pre[iot_index, edge_index] = operate.z_pre[iot_index, edge_index]

    def choose_action(self):
        start_time = time.time()
        res = self.opt.solver.solve(self.opt.model)
        # res.write()

        end_time = time.time()

        solution_time = end_time - start_time

        opt_val = value(self.opt.model.obj)

        offloads: np.ndarray = np.zeros([len(self.opt.model.I), len(self.opt.model.J)])
        actions = [0 for _ in self.opt.model.I]
        actions = [int(action) for action in actions]

        for iot_index in range(len(self.opt.model.I)):
            for edge_index in range(len(self.opt.model.J)):
                if value(self.opt.model.x[iot_index, edge_index]) == 1:
                    offloads[iot_index, edge_index] = 1
                    actions[iot_index] = edge_index + 1

        allocation: List = list()
        if self.is_scheduled:
            for edge in range(len(self.opt.model.J)):
                allocation.append(np.zeros([len(self.opt.model.I), len(self.opt.model.I)]))

            for edge_index in range(len(self.opt.model.J)):
                for iot_index in range(len(self.opt.model.I)):
                    for iot_index_ in range(len(self.opt.model.I)):
                        if value(self.opt.model.k[edge_index, iot_index, iot_index_]) == 1:
                            allocation[edge_index][iot_index, iot_index_] = 1

        container_caching: np.ndarray = np.zeros([len(self.opt.model.I), len(self.opt.model.J)])
        image_caching: np.ndarray = np.zeros([len(self.opt.model.I), len(self.opt.model.J)])
        if self.has_caching:
            for iot_index in range(len(self.opt.model.I)):
                for edge_index in range(len(self.opt.model.J)):

                    if value(self.opt.model.y[iot_index, edge_index]) == 1:
                        container_caching[iot_index, edge_index] = 1
                    else:
                        container_caching[iot_index, edge_index] = 0

                    if value(self.opt.model.z[iot_index, edge_index]) == 1:
                        image_caching[iot_index, edge_index] = 1
                    else:
                        image_caching[iot_index, edge_index] = 0

        # self.opt.display_solution()
        # self.opt.display_linear_coe()

        if self.is_scheduled and self.has_caching:
            # deletion print obj value
            if self.deletion:
                obj_value = self.get_obj_value(x=offloads, y=container_caching, z=image_caching, k=allocation)
                print(f'obj_value = {obj_value}')
            return actions, allocation, container_caching, image_caching, solution_time, opt_val
        elif self.is_scheduled:
            return actions, allocation, solution_time, opt_val
        elif self.has_caching:
            return actions, container_caching, image_caching, solution_time, opt_val
        else:
            return actions, solution_time, opt_val

    def get_obj_value(self, x, y, z, k):
        obj_value = 0
        for i in self.opt.model.I:
            # mod.comp_time_on_iot[i]
            obj_value += (1 - sum(x[i, j] for j in self.opt.model.J)) * (
                        value(self.opt.model.G[i]) + value(self.opt.model.D[i])) * value(self.opt.model.rho[i]) / value(
                self.opt.model.Fd[i])
            # mod.tran_time_on_iot[i]
            obj_value += sum(
                x[i, j] * (value(self.opt.model.U[i, j]) + value(self.opt.model.D[i])) / value(self.opt.model.R[i, j]) for j in
                self.opt.model.J)
            # prepare
            obj_value += sum(
                x[i, j] * y[i, j] * (1 - value(self.opt.model.y_pre[i, j])) * value(self.opt.model.theta[i])
                for j in self.opt.model.J)
            obj_value += sum(
                x[i, j] * z[i, j] * (1 - value(self.opt.model.z_pre[i, j])) * value(self.opt.model.mu[i]) / value(self.opt.model.R_pull[j])
                for j in self.opt.model.J)
            # wait
            obj_value += sum(sum(
                x[i, j] * value(self.opt.model.Q[ii, j]) * value(self.opt.model.rho[ii]) / value(self.opt.model.Fe[j]) +
                x[i, j] * x[ii, j] * k[j][ii, i] * value(self.opt.model.D[ii]) * self.opt.model.rho[ii] / self.opt.model.Fe[j]
                for ii in self.opt.model.I)
                             for j in self.opt.model.J)

            # edge computing time
            obj_value += sum(
                x[i, j] * value(self.opt.model.D[i]) * value(self.opt.model.rho[i]) / value(self.opt.model.Fe[j])
                for j in self.opt.model.J)

            # mod.delete_time[i, j]
            obj_value += sum(
                x[i, j] * (1 - y[i, j]) * value(self.opt.model.y_pre[i, j]) * value(self.opt.model.delta_container[i]) for j in
                self.opt.model.J)
            obj_value += sum(
                x[i, j] * (1 - z[i, j]) * value(self.opt.model.z_pre[i, j]) * value(self.opt.model.delta_image[i]) for j in
                self.opt.model.J)

            # plus deletion
            obj_value += sum(self.opt.model.delta_container[i] * (y[i, j] - value(self.opt.model.y_pre[i, j])) ** 2
                             + self.opt.model.delta_image[i] * (z[i, j] - value(self.opt.model.z_pre[i, j])) ** 2for j in self.opt.model.J)

        # print model obj value
        model_obj = sum(self.opt.model.coe_non[i] +
               sum(sum(self.opt.model.coe_q[i_, i, j] * x[i, j] * x[i_, j] * k[j][i_, i] for i_ in self.opt.model.I) for j in self.opt.model.J) +
               sum(self.opt.model.coe_x[i, j] * x[i, j] for j in self.opt.model.J)
               + sum(self.opt.model.coe_l[i, j] * x[i, j] * y[i, j] for j in self.opt.model.J)
               + sum(self.opt.model.coe_m[i, j] * x[i, j] * z[i, j] for j in self.opt.model.J)
               + sum(self.opt.model.delta_container[i] * (y[i, j] - value(self.opt.model.y_pre[i, j])) ** 2
                     + self.opt.model.delta_image[i] * (z[i, j] - value(self.opt.model.z_pre[i, j])) ** 2 for j in self.opt.model.J)
               for i in self.opt.model.I)
        # print(f'model_obj_value = {value(model_obj)}')

        return obj_value
