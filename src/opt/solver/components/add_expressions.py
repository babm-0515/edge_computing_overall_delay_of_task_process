from pyomo.environ import Expression


def add_basic_expressions(mod):

    def iot_comp_delay_rule(md, i):
        return (md.G[i] + md.D[i]) * md.rho[i] / md.Fd[i]
    mod.t_l = Expression(mod.I, rule=iot_comp_delay_rule)

    def iot_tran_delay_rule(md, i, j):
        return (md.U[i, j] + md.D[i]) / md.R[i, j]
    mod.t_u = Expression(mod.I, mod.J, rule=iot_tran_delay_rule)

    def container_start_delay_rule(md, i, j):
        return (1 - md.y_pre[i, j]) * md.theta[i]
    mod.t_s = Expression(mod.I, mod.J, rule=container_start_delay_rule)

    def image_deploy_delay_rule(md, i, j):
        return (1 - md.z_pre[i, j]) * md.mu[i] / md.R_pull[j]
    mod.t_d = Expression(mod.I, mod.J, rule=image_deploy_delay_rule)

    def container_delete_delay_rule(md, i, j):
        return md.y_pre[i, j] * md.delta_container[i]
    mod.t_c = Expression(mod.I, mod.J, rule=container_delete_delay_rule)

    def image_delete_delay_rule(md, i, j):
        return md.z_pre[i, j] * md.delta_image[i]
    mod.t_m = Expression(mod.I, mod.J, rule=image_delete_delay_rule)

    def edge_remain_delay_rule(md, j):
        return sum(md.Q[i_, j] * md.rho[i_] / md.Fe[j] for i_ in md.I)
    mod.t_r = Expression(mod.J, rule=edge_remain_delay_rule)

    def edge_process_delay_rule(md, i, j):
        return md.D[i] * md.rho[i] / md.Fe[j]
    mod.t_e = Expression(mod.I, mod.J, rule=edge_process_delay_rule)


def offload_object_rule(md):
    return sum(md.t_l[i] + sum((md.t_u[i, j] + md.t_e[i, j] - md.t_l[i]) * md.x[i, j] for j in md.J)
               for i in md.I)


def cache_object_rule(md):
    return sum(md.t_l[i] + sum((md.t_u[i, j] + md.t_e[i, j] - md.t_l[i] + md.t_c[i, j] + md.t_m[i, j]) * md.x[i, j]
                               for j in md.J)
               + sum((md.t_s[i, j] - md.t_c[i, j]) * md.il[i, j] for j in md.J)
               + sum((md.t_d[i, j] - md.t_m[i, j]) * md.im[i, j]for j in md.J)
               for i in md.I)


def schedule_object_rule(md):
    return sum(md.t_l[i] + sum((md.t_u[i, j] + md.t_r[j] + md.t_e[i, j] - md.t_l[i]) * md.x[i, j] for j in md.J)
               + sum(sum(md.t_e[i_, j] * md.iq[j, i_, i] for i_ in md.I) for j in md.J)
               for i in md.I)


def nonlinear_object_rule(md):
    # noinspection DuplicatedCode
    return sum(md.t_l[i] + sum(sum(md.t_e[i_, j] * md.k[j, i_, i] * md.x[i_, j] * md.x[i, j] for i_ in md.I)
                               for j in md.J)
               + sum((md.t_u[i, j] + md.t_r[j] + md.t_e[i, j] - md.t_l[i] + md.t_c[i, j] + md.t_m[i, j]) * md.x[i, j]
                     for j in md.J)
               + sum((md.t_s[i, j] - md.t_c[i, j]) * md.x[i, j] * md.y[i, j] for j in md.J)
               + sum((md.t_d[i, j] - md.t_m[i, j]) * md.x[i, j] * md.z[i, j] for j in md.J)
               for i in md.I)


def linear_object_rule(md):
    # noinspection DuplicatedCode
    return sum(md.coe_non[i] +
               sum(sum(md.coe_q[i_, i, j] * md.iq[j, i_, i] for i_ in md.I) for j in md.J)
               + sum(md.coe_x[i, j] * md.x[i, j] for j in md.J)
               + sum(md.coe_l[i, j] * md.il[i, j] for j in md.J)
               + sum(md.coe_m[i, j] * md.im[i, j] for j in md.J)
               for i in md.I)


def linear_plus_deletion_object_rule(md):
    # noinspection DuplicatedCode
    return sum(md.coe_non[i] +
               sum(sum(md.coe_q[i_, i, j] * md.iq[j, i_, i] for i_ in md.I) for j in md.J)
               + sum(md.coe_x[i, j] * md.x[i, j] for j in md.J)
               + sum(md.coe_l[i, j] * md.il[i, j] for j in md.J)
               + sum(md.coe_m[i, j] * md.im[i, j] for j in md.J)
               + sum(md.delta_container[i] * md.del_y[i, j] + md.delta_image[i] * md.del_z[i, j] for j in md.J)
               for i in md.I)
