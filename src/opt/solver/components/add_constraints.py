from pyomo.environ import Constraint


def offload_cons_rule(mod, i):
    return sum(mod.x[i, j] for j in mod.J) <= 1


def add_cache_cons(mod):
    def cache_con1_rule(md, i, j):
        return md.x[i, j] <= md.y[i, j]
    mod.cache_con1 = Constraint(mod.I, mod.J, rule=cache_con1_rule)

    def cache_con2_rule(md, i, j):
        return md.y[i, j] * (1 - md.y_pre[i, j]) <= md.z[i, j]
    mod.cache_con2 = Constraint(mod.I, mod.J, rule=cache_con2_rule)

    def cache_con3_rule(md, i, j):
        up_bound = 1000
        return md.Q[i, j] <= up_bound * md.y[i, j]
    mod.cache_con3 = Constraint(mod.I, mod.J, rule=cache_con3_rule)

    def cache_con4_rule(md, j):
        return sum(md.y[i, j] * md.eta[i] for i in md.I) <= md.eta_max[j]
    mod.cache_con4 = Constraint(mod.J, rule=cache_con4_rule)

    def cache_con5_rule(md, j):
        return sum(md.z[i, j] * md.mu[i] for i in md.I) <= md.mu_max[j]
    mod.cache_con5 = Constraint(mod.J, rule=cache_con5_rule)


def add_schedule_cons(mod):
    def schedule_con1_rule(md, j, i, ii):
        if i != ii:
            return md.k[j, i, ii] + md.k[j, ii, i] == 1
        return Constraint.Skip
    mod.schedule_con1 = Constraint(mod.J, mod.I, mod.I, rule=schedule_con1_rule)

    def schedule_con2_rule(md, j, i, ii, iii):
        if i != ii and ii != iii and i != iii:
            return md.k[j, i, ii] + md.k[j, ii, iii] <= 1 + md.k[j, i, iii]
        return Constraint.Skip
    mod.schedule_con2 = Constraint(mod.J, mod.I, mod.I, mod.I, rule=schedule_con2_rule)

    def schedule_con3_rule(md, j, i):
        return md.k[j, i, i] == 0
    mod.schedule_con3 = Constraint(mod.J, mod.I, rule=schedule_con3_rule)


def add_deletion_var_cons(mod):
    # plus deletion cons
    def del_y_c1_rule(md, i, j):
        return md.del_y[i, j] >= md.y[i, j] - md.y_pre[i, j]
    mod.del_y_c1 = Constraint(mod.I, mod.J, rule=del_y_c1_rule)

    def del_y_c2_rule(md, i, j):
        return md.del_y[i, j] >= md.y_pre[i, j] - md.y[i, j]
    mod.del_y_c2 = Constraint(mod.I, mod.J, rule=del_y_c2_rule)

    def del_y_c3_rule(md, i, j):
        return md.del_y[i, j] <= md.y[i, j] + md.y_pre[i, j]
    mod.del_y_c3 = Constraint(mod.I, mod.J, rule=del_y_c3_rule)

    def del_y_c4_rule(md, i, j):
        return md.del_y[i, j] <= 2 - (md.y[i, j] + md.y_pre[i, j])
    mod.del_y_c4 = Constraint(mod.I, mod.J, rule=del_y_c4_rule)

    def del_z_c1_rule(md, i, j):
        return md.del_z[i, j] >= md.z[i, j] - md.z_pre[i, j]
    mod.del_z_c1 = Constraint(mod.I, mod.J, rule=del_z_c1_rule)

    def del_z_c2_rule(md, i, j):
        return md.del_z[i, j] >= md.z_pre[i, j] - md.z[i, j]
    mod.del_z_c2 = Constraint(mod.I, mod.J, rule=del_z_c2_rule)

    def del_z_c3_rule(md, i, j):
        return md.del_z[i, j] <= md.z[i, j] + md.z_pre[i, j]
    mod.del_z_c3 = Constraint(mod.I, mod.J, rule=del_z_c3_rule)

    def del_z_c4_rule(md, i, j):
        return md.del_z[i, j] <= 2 - (md.z[i, j] + md.z_pre[i, j])
    mod.del_z_c4 = Constraint(mod.I, mod.J, rule=del_z_c4_rule)


def _add_aux_var_cons(mod, var1, var2, var3, index1, index2, name_prefix):
    """var1 = var2 * var3"""
    def c1_rule(_, i, j):
        return var1[i, j] <= var2[i, j]

    def c2_rule(_, i, j):
        return var1[i, j] <= var3[i, j]

    def c3_rule(_, i, j):
        return var2[i, j] + var3[i, j] - 1 <= var1[i, j]

    setattr(mod, f"{name_prefix}_c1", Constraint(index1, index2, rule=c1_rule))
    setattr(mod, f"{name_prefix}_c2", Constraint(index1, index2, rule=c2_rule))
    setattr(mod, f"{name_prefix}_c3", Constraint(index1, index2, rule=c3_rule))


def add_aux_var_lm_cons(mod):
    _add_aux_var_cons(mod, mod.il, mod.x, mod.y, mod.I, mod.J, "il")
    _add_aux_var_cons(mod, mod.im, mod.x, mod.z, mod.I, mod.J, "im")


def add_aux_var_pq_cons(mod):
    def ip_c1_rule(md, j, ii, i):
        return md.ip[j, ii, i] <= md.k[j, ii, i]
    mod.ip_c1 = Constraint(mod.J, mod.I, mod.I, rule=ip_c1_rule)

    def ip_c2_rule(md, j, ii, i):
        return md.ip[j, ii, i] <= md.x[ii, j]
    mod.ip_c2 = Constraint(mod.J, mod.I, mod.I, rule=ip_c2_rule)

    def ip_c3_rule(md, j, ii, i):
        return md.k[j, ii, i] + md.x[ii, j] - 1 <= md.ip[j, ii, i]
    # noinspection DuplicatedCode
    mod.ip_c3 = Constraint(mod.J, mod.I, mod.I, rule=ip_c3_rule)

    def iq_c1_rule(md, j, ii, i):
        return md.iq[j, ii, i] <= md.x[i, j]
    mod.iq_c1 = Constraint(mod.J, mod.I, mod.I, rule=iq_c1_rule)

    def iq_c2_rule(md, j, ii, i):
        return md.iq[j, ii, i] <= md.ip[j, ii, i]
    mod.iq_c2 = Constraint(mod.J, mod.I, mod.I, rule=iq_c2_rule)

    def iq_c3_rule(md, j, ii, i):
        return md.x[i, j] + md.ip[j, ii, i] - 1 <= md.iq[j, ii, i]
    mod.iq_c3 = Constraint(mod.J, mod.I, mod.I, rule=iq_c3_rule)

