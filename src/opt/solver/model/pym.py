import numpy as np
from pyomo.environ import Binary, SolverFactory, ConcreteModel, Set, Var, Param, Constraint, Objective, value
from pyomo.environ import minimize, Expression
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr import MonomialTermExpression

from src.scenario.env import Env
import src.opt.solver.components.add_constraints as cons
import src.opt.solver.components.add_expressions as exps


class PyoModel:
    def __init__(self, env: Env, solver: str = 'scip'):
        self.model = ConcreteModel()
        self.solver = SolverFactory(solver)

        self.model.I = Set(initialize=[i for i in range(env.n_iot)])    # noqa: E741
        self.model.J = Set(initialize=[j for j in range(env.n_edge)])

        self.model.x = Var(self.model.I, self.model.J, within=Binary)

        self.model.Fd = Param(self.model.I, initialize=env.comp_cap_iot)
        self.model.Fe = Param(self.model.J, initialize=env.comp_cap_edge)
        self.model.rho = Param(self.model.I, initialize=env.comp_density)
        tran_cap_dict = {(i, j): env.tran_cap_iot[i][j] for i in self.model.I for j in self.model.J}
        self.model.R = Param(self.model.I, self.model.J, initialize=tran_cap_dict)

        self.model.theta = Param(self.model.I, initialize=env.container_startup)
        self.model.eta = Param(self.model.I, initialize=env.container_size)
        self.model.delta_container = Param(self.model.I, initialize=env.container_delete)

        self.model.mu = Param(self.model.I, initialize=env.image_size)
        self.model.delta_image = Param(self.model.I, initialize=env.image_delete)
        self.model.R_pull = Param(self.model.J, initialize=env.edge_download_speed)

        self.model.eta_max = Param(self.model.J, initialize=env.edge_container_cache_limit)
        self.model.mu_max = Param(self.model.J, initialize=env.edge_image_cache_limit)

        self.model.D = Param(self.model.I, initialize=np.zeros(env.n_iot), mutable=True)
        self.model.G = Param(self.model.I, initialize=np.zeros(env.n_iot), mutable=True)
        self.model.U = Param(self.model.I, self.model.J, initialize={(i, j): 0 for i in self.model.I for j in self.model.J}, mutable=True)
        self.model.Q = Param(self.model.I, self.model.J, initialize={(i, j): 0 for i in self.model.I for j in self.model.J}, mutable=True)

        self.model.y_pre = Param(self.model.I, self.model.J, initialize={(i, j): 0 for i in self.model.I for j in self.model.J}, mutable=True)
        self.model.z_pre = Param(self.model.I, self.model.J, initialize={(i, j): 0 for i in self.model.I for j in self.model.J}, mutable=True)

        self.model.offload_cons = Constraint(self.model.I, rule=cons.offload_cons_rule)
        exps.add_basic_expressions(self.model)

    def add_cache_var_cons(self):
        self.model.y = Var(self.model.I, self.model.J, within=Binary)
        self.model.z = Var(self.model.I, self.model.J, within=Binary)
        cons.add_cache_cons(self.model)

    def add_schedule_var_cons(self):
        self.model.k = Var(self.model.J, self.model.I, self.model.I, within=Binary)
        cons.add_schedule_cons(self.model)

    def add_deletion_var_cons(self):
        # plus deletion var
        self.model.del_y = Var(self.model.I, self.model.J, within=Binary)
        self.model.del_z = Var(self.model.I, self.model.J, within=Binary)
        cons.add_deletion_var_cons(self.model)

    def set_offload_object(self):
        self.model.obj = Objective(sense=minimize, rule=exps.offload_object_rule)

    def set_cache_object(self):
        self._add_aux_var_lm_cons()
        self.model.obj = Objective(sense=minimize, rule=exps.cache_object_rule)

    def set_schedule_object(self):
        self._add_aux_var_pq_cons()
        self.model.obj = Objective(sense=minimize, rule=exps.schedule_object_rule)

    def set_nonlinear_object(self):
        self.model.obj = Objective(sense=minimize, rule=exps.nonlinear_object_rule)

    def set_linear_object(self):
        self._add_aux_var_lm_cons()
        self._add_aux_var_pq_cons()

        def coe_non_rule(md, i):
            return md.t_l[i]
        self.model.coe_non = Expression(self.model.I, rule=coe_non_rule)

        def coe_q_rule(md, i_, i, j):
            return md.t_e[i_, j]
        self.model.coe_q = Expression(self.model.I, self.model.I, self.model.J, rule=coe_q_rule)

        def coe_x_rule(md, i, j):
            return md.t_u[i, j] + md.t_r[j] + md.t_e[i, j] - md.t_l[i] + md.t_c[i, j] + md.t_m[i, j]
        self.model.coe_x = Expression(self.model.I, self.model.J, rule=coe_x_rule)

        def coe_l_rule(md, i, j):
            return md.t_s[i, j] - md.t_c[i, j]
        self.model.coe_l = Expression(self.model.I, self.model.J, rule=coe_l_rule)

        def coe_m_rule(md, i, j):
            return md.t_d[i, j] - md.t_m[i, j]
        self.model.coe_m = Expression(self.model.I, self.model.J, rule=coe_m_rule)

        self.model.obj = Objective(sense=minimize, rule=exps.linear_object_rule)

    def set_linear_plus_deletion_object(self):
        self._add_aux_var_lm_cons()
        self._add_aux_var_pq_cons()

        def coe_non_rule(md, i):
            return md.t_l[i]

        self.model.coe_non = Expression(self.model.I, rule=coe_non_rule)

        def coe_q_rule(md, i_, i, j):
            return md.t_e[i_, j]

        self.model.coe_q = Expression(self.model.I, self.model.I, self.model.J, rule=coe_q_rule)

        def coe_x_rule(md, i, j):
            return md.t_u[i, j] + md.t_r[j] + md.t_e[i, j] - md.t_l[i] + md.t_c[i, j] + md.t_m[i, j]

        self.model.coe_x = Expression(self.model.I, self.model.J, rule=coe_x_rule)

        def coe_l_rule(md, i, j):
            return md.t_s[i, j] - md.t_c[i, j]

        self.model.coe_l = Expression(self.model.I, self.model.J, rule=coe_l_rule)

        def coe_m_rule(md, i, j):
            return md.t_d[i, j] - md.t_m[i, j]

        self.model.coe_m = Expression(self.model.I, self.model.J, rule=coe_m_rule)

        # plus deletion obj
        self.add_deletion_var_cons()
        self.model.obj = Objective(sense=minimize, rule=exps.linear_plus_deletion_object_rule)

    def display_linear_coe(self):
        print(f'display_linear_coe: ')
        for i in self.model.I:
            print(f'coe_non{i}={value(self.model.coe_non[i])}')
        for i_ in self.model.I:
            for i in self.model.I:
                for j in self.model.J:
                    print(f'coe_q{i_}_{i}_{j}={value(self.model.coe_q[i_, i, j])}')
        for i in self.model.I:
            for j in self.model.J:
                print(f'coe_x{i}_{j}={value(self.model.coe_x[i, j])}')
        for i in self.model.I:
            for j in self.model.J:
                print(f'coe_l{i}_{j}={value(self.model.coe_l[i, j])}')
        for i in self.model.I:
            for j in self.model.J:
                print(f'coe_m{i}_{j}={value(self.model.coe_m[i, j])}')

    def _add_aux_var_lm_cons(self):
        self.model.il = Var(self.model.I, self.model.J, within=Binary)
        self.model.im = Var(self.model.I, self.model.J, within=Binary)
        cons.add_aux_var_lm_cons(self.model)

    def _add_aux_var_pq_cons(self):
        self.model.ip = Var(self.model.J, self.model.I, self.model.I, within=Binary)
        self.model.iq = Var(self.model.J, self.model.I, self.model.I, within=Binary)
        cons.add_aux_var_pq_cons(self.model)

    def display_solution(self):
        # print(f'obj = {self.model.obj.expr}')
        for i in self.model.I:
            print(f't_l_{i}={value(self.model.t_l[i])}')
        for i in self.model.I:
            for j in self.model.J:
                print(f't_u_{i}_{j}={value(self.model.t_u[i, j])}')
        for i in self.model.I:
            for j in self.model.J:
                print(f't_s_{i}_{j}={value(self.model.t_s[i, j])}')
        for i in self.model.I:
            for j in self.model.J:
                print(f't_d_{i}_{j}={value(self.model.t_d[i, j])}')
        for i in self.model.I:
            for j in self.model.J:
                print(f't_c_{i}_{j}={value(self.model.t_c[i, j])}')
        for i in self.model.I:
            for j in self.model.J:
                print(f't_m_{i}_{j}={value(self.model.t_m[i, j])}')
        for j in self.model.J:
            print(f't_r_{j}={value(self.model.t_r[j])}')
        for i in self.model.I:
            for j in self.model.J:
                print(f't_e_{i}_{j}={value(self.model.t_e[i, j])}')

        for i in self.model.I:
            for j in self.model.J:
                if value(self.model.x[i, j]) != 0:
                    print(f'x_{i}_{j}={value(self.model.x[i, j])}')
        for i in self.model.I:
            for j in self.model.J:
                if value(self.model.y[i, j]) != 0:
                    print(f'y_{i}_{j}={value(self.model.y[i, j])}')
        for i in self.model.I:
            for j in self.model.J:
                if value(self.model.z[i, j]) != 0:
                    print(f'z_{i}_{j}={value(self.model.z[i, j])}')
        for i_ in self.model.I:
            for i in self.model.I:
                for j in self.model.J:
                    if value(self.model.k[j, i_, i]) != 0:
                        print(f'k_{i_}_{i}_{j}={value(self.model.k[j, i_, i])}')

