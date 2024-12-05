import numpy as np
from collections import deque

from src.scenario.operate import Operate


class Sim:
    def __init__(self, num_iot, num_edge, num_time, duration, task_arrive_prob, max_bit_arrive, min_bit_arrive,
                 comp_cap_iot, comp_cap_edge, tran_cap_iot, comp_density, container_size, container_startup,
                 container_delete, image_size, image_delete, edge_container_cache_limit, edge_image_cache_limit,
                 edge_download_speed):
        self.operate = Operate(num_iot, num_edge, num_time, duration, task_arrive_prob, max_bit_arrive, min_bit_arrive,
                               comp_cap_iot, comp_cap_edge, tran_cap_iot, comp_density, container_size, container_startup,
                               container_delete, image_size, image_delete, edge_container_cache_limit,
                               edge_image_cache_limit, edge_download_speed)

    def reset(self, bit_arrive):
        self.operate.env.bit_arrive = bit_arrive

        # TIME COUNT
        self.operate.env.time_count = int(0)

        self.operate.env.Queue_iot_comp = list()
        self.operate.env.Queue_iot_tran = list()
        self.operate.env.Queue_edge_wait = list()
        self.operate.env.Queue_edge_comp = list()

        for iot in range(self.operate.env.n_iot):
            self.operate.env.Queue_iot_comp.append(deque())
            self.operate.env.Queue_iot_tran.append(list())
            self.operate.env.Queue_edge_wait.append(list())
            for edge in range(self.operate.env.n_edge):
                self.operate.env.Queue_iot_tran[iot].append(deque())
                self.operate.env.Queue_edge_wait[iot].append(deque())
        for edge in range(self.operate.env.n_edge):
            self.operate.env.Queue_edge_comp.append(deque())

        # TASK DELAY
        self.operate.env.process_delay = np.zeros([self.operate.env.n_time, self.operate.env.n_iot])
        self.operate.env.process_delay_trans = np.zeros([self.operate.env.n_time, self.operate.env.n_iot])

        self.operate.update_info_for_opt()

    def _find_task_order(self, k):
        # Step 1: Compute the in-degree of each node
        in_degree = [int(sum(k[ii, i] for ii in range(self.operate.env.n_iot))) for i in range(self.operate.env.n_iot)]

        # Step 3: Process the queue and perform the topological sort
        order = [0 for i in range(self.operate.env.n_iot)]

        for i in range(self.operate.env.n_iot):
            order[in_degree[i]] = i

        # display order
        # print(f'k = {k}, order = {order}')

        return order

    def _process_env(self, schedule=None, container_cache=None, image_cache=None):
        # PROCESS IOT COMP AND TRANS QUEUE
        self.operate.process_iot_comp_queue()
        self.operate.process_iot_tran_queue()

        # PROCESS EDGE
        for edge_index in range(self.operate.env.n_edge):
            if schedule is None:
                self.operate.put_task_into_edge_comp_in_circle(edge_index=edge_index)
            else:
                task_order = self._find_task_order(schedule[edge_index])
                self.operate.put_task_into_edge_comp_in_order(edge_index=edge_index, order=task_order)

            if container_cache is None or image_cache is None:
                self.operate.process_edge_comp_without_caching(edge_index)
            else:
                self.operate.process_edge_comp_with_caching(edge_index, container_cache, image_cache)

        # TIME UPDATE
        self.operate.env.time_count += 1

    def step(self, offload=None, schedule=None, container_cache=None, image_cache=None):
        # GET OFFLOAD ACTIONS
        iot_action_local: np.ndarray = np.zeros([self.operate.env.n_iot], np.int32)
        iot_action_edge: np.ndarray = np.zeros([self.operate.env.n_iot], np.int32)
        for iot_index in range(self.operate.env.n_iot):
            iot_action = offload[iot_index]
            iot_action_edge[iot_index] = int(iot_action - 1)
            if iot_action == 0:
                iot_action_local[iot_index] = 1

        # PUT TASKS INTO IOT QUEUE
        for iot_index in range(self.operate.env.n_iot):
            if self.operate.env.bit_arrive[self.operate.env.time_count, iot_index] != 0:
                if iot_action_local[iot_index] == 1:
                    self.operate.put_task_into_iot_comp(iot_index=iot_index)
                else:
                    self.operate.put_task_into_iot_trans(iot_index=iot_index, edge_index=iot_action_edge[iot_index])

        self._process_env(schedule, container_cache, image_cache)

        done = False
        if self.operate.env.time_count >= self.operate.env.n_time:
            while not done:
                queue_empty = True
                for iot_index in range(self.operate.env.n_iot):
                    if len(self.operate.env.Queue_iot_comp[iot_index]) > 0:
                        queue_empty = False
                        break
                    for edge_index in range(self.operate.env.n_edge):
                        if len(self.operate.env.Queue_iot_tran[iot_index][edge_index]) > 0 \
                                or len(self.operate.env.Queue_edge_wait[iot_index][edge_index]) > 0:
                            queue_empty = False
                            break

                for edge_index in range(self.operate.env.n_edge):
                    if len(self.operate.env.Queue_edge_comp[edge_index]) > 0:
                        queue_empty = False
                        break

                if queue_empty:
                    done = True
                else:
                    self._process_env(schedule, container_cache, image_cache)

        self.operate.update_info_for_opt()

        return done
