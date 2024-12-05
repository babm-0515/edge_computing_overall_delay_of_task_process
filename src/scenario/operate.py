import numpy as np

from src.scenario.env import Env


class Operate:
    def __init__(self, num_iot, num_edge, num_time, duration, task_arrive_prob, max_bit_arrive, min_bit_arrive,
                 comp_cap_iot, comp_cap_edge, tran_cap_iot, comp_density, container_size, container_startup,
                 container_delete, image_size, image_delete, edge_container_cache_limit, edge_image_cache_limit,
                 edge_download_speed):
        self.env = Env(num_iot, num_edge, num_time, duration, task_arrive_prob, max_bit_arrive, min_bit_arrive,
                       comp_cap_iot, comp_cap_edge, tran_cap_iot, comp_density, container_size, container_startup,
                       container_delete, image_size, image_delete, edge_container_cache_limit, edge_image_cache_limit,
                       edge_download_speed)
        self.is_serial = True

        # INFO FOR OPTIMIZATION
        self.queue_iot_comp_remain: np.ndarray = np.zeros(self.env.n_iot)
        self.queue_iot_tran_remain: np.ndarray = np.zeros([self.env.n_iot, self.env.n_edge])
        self.queue_fog_comp_remain: np.ndarray = np.zeros([self.env.n_iot, self.env.n_edge])
        self.y_pre: np.ndarray = np.zeros([self.env.n_iot, self.env.n_edge])
        self.z_pre: np.ndarray = np.zeros([self.env.n_iot, self.env.n_edge])

    def put_task_into_iot_comp(self, iot_index):
        bit_arrive = np.squeeze(self.env.bit_arrive[self.env.time_count, iot_index])
        iot_comp_task = {'time': self.env.time_count, 'size': bit_arrive, 'remain': bit_arrive}
        self.env.Queue_iot_comp[iot_index].append(iot_comp_task)

    def put_task_into_iot_trans(self, iot_index, edge_index):
        bit_arrive = np.squeeze(self.env.bit_arrive[self.env.time_count, iot_index])
        iot_tran_task = {'time': self.env.time_count, 'edge': edge_index,
                         'size': bit_arrive, 'remain': bit_arrive}
        self.env.Queue_iot_tran[iot_index][iot_tran_task['edge']].append(iot_tran_task)

    def process_iot_comp_queue(self):
        for iot_index in range(self.env.n_iot):
            iot_comp_cap = self.env.comp_cap_iot[iot_index]
            iot_comp_density = self.env.comp_density[iot_index]

            iot_comp_time_remain = self.env.duration

            while iot_comp_time_remain > 0 and len(self.env.Queue_iot_comp[iot_index]) > 0:
                get_task = self.env.Queue_iot_comp[iot_index].popleft()

                if iot_comp_time_remain >= get_task['remain'] * iot_comp_density / iot_comp_cap:
                    iot_comp_time_remain -= get_task['remain'] * iot_comp_density / iot_comp_cap
                    self.env.process_delay[get_task['time'], iot_index] = \
                        self.env.time_count * self.env.duration + (self.env.duration - iot_comp_time_remain) - get_task[
                            'time'] * self.env.duration
                else:
                    get_task['remain'] -= iot_comp_time_remain * iot_comp_cap / iot_comp_density
                    iot_comp_time_remain = 0
                    self.env.Queue_iot_comp[iot_index].appendleft(get_task)

    def process_iot_tran_queue(self):
        for iot_index in range(self.env.n_iot):
            for edge_index in range(self.env.n_edge):
                iot_tran_cap = self.env.tran_cap_iot[iot_index, edge_index]

                iot_tran_time_remain = self.env.duration

                while iot_tran_time_remain > 0 and len(self.env.Queue_iot_tran[iot_index][edge_index]) != 0:
                    get_task = self.env.Queue_iot_tran[iot_index][edge_index].popleft()

                    if iot_tran_time_remain >= get_task['remain'] / iot_tran_cap:
                        iot_tran_time_remain -= get_task['remain'] / iot_tran_cap

                        tmp_dict = {'iot': iot_index, 'time': get_task['time'],
                                    'size': get_task['size'], 'remain': get_task['size']}
                        self.env.Queue_edge_wait[iot_index][edge_index].append(tmp_dict)

                        self.env.process_delay_trans[get_task['time'], iot_index] \
                            = self.env.time_count * self.env.duration + (self.env.duration - iot_tran_time_remain) - \
                            get_task['time'] * self.env.duration

                    else:
                        get_task['remain'] -= iot_tran_time_remain * iot_tran_cap
                        iot_tran_time_remain = 0
                        self.env.Queue_iot_tran[iot_index][edge_index].appendleft(get_task)

    def put_task_into_edge_comp_in_circle(self, edge_index):
        for iot_index in range(self.env.n_iot):
            while len(self.env.Queue_edge_wait[iot_index][edge_index]) > 0:
                get_task = self.env.Queue_edge_wait[iot_index][edge_index].popleft()
                self.env.Queue_edge_comp[edge_index].append(get_task)

    def put_task_into_edge_comp_in_order(self, edge_index, order):
        for iot_index in order:
            while len(self.env.Queue_edge_wait[iot_index][edge_index]) > 0:
                get_task = self.env.Queue_edge_wait[iot_index][edge_index].popleft()
                self.env.Queue_edge_comp[edge_index].append(get_task)

    def process_edge_comp_without_caching(self, edge_index):
        edge_time_remain = self.env.duration

        if self.is_serial is True:
            self._process_edge_comp(edge_time_remain, edge_index)
        else:
            self._process_edge_comp_in_concurrent(edge_time_remain, edge_index)

    def process_edge_comp_with_caching(self, edge_index, container_caching, image_caching):
        edge_time_remain = self.env.duration

        for iot_index in range(self.env.n_iot):
            if self.env.edge_server_cache[edge_index].has_image(image_index=iot_index):
                if image_caching[iot_index][edge_index] == 0:
                    edge_time_remain -= self.env.edge_server_cache[edge_index].del_image(image_index=iot_index)

            if self.env.edge_server_cache[edge_index].has_container(container_index=iot_index):
                if container_caching[iot_index][edge_index] == 0:
                    edge_time_remain -= self.env.edge_server_cache[edge_index].del_container(container_index=iot_index)

        if self.is_serial is True:
            self._process_edge_comp(edge_time_remain, edge_index)
        else:
            self._process_edge_comp_in_concurrent(edge_time_remain, edge_index)

    def _process_edge_comp(self, edge_time_remain, edge_index):
        comp_cap_edge = self.env.comp_cap_edge[edge_index]
        while edge_time_remain > 0 and len(self.env.Queue_edge_comp[edge_index]) > 0:
            get_task = self.env.Queue_edge_comp[edge_index].popleft()

            iot_index = get_task['iot']
            iot_comp_density = self.env.comp_density[iot_index]
            edge_time_remain -= self.env.edge_server_cache[edge_index].dep_container(container_index=iot_index)

            if self.env.edge_server_cache[edge_index].has_container(container_index=iot_index):
                if edge_time_remain * comp_cap_edge / iot_comp_density >= get_task['remain']:
                    edge_time_remain -= get_task['remain'] * iot_comp_density / comp_cap_edge
                    self.env.process_delay[get_task['time'], iot_index] = self.env.time_count * self.env.duration + \
                        (self.env.duration - edge_time_remain) - get_task['time'] * self.env.duration

                elif edge_time_remain * comp_cap_edge / iot_comp_density < get_task['remain']:
                    get_task['remain'] -= edge_time_remain * comp_cap_edge / iot_comp_density
                    edge_time_remain = 0
                    self.env.Queue_edge_comp[edge_index].appendleft(get_task)

            else:
                self.env.Queue_edge_comp[edge_index].appendleft(get_task)

    def _activated_comp_task(self, edge_index):
        activated_queues = {}
        for task in self.env.Queue_edge_comp[edge_index]:
            if task['remain'] > 0:
                activated_queues[task['iot']] = activated_queues.get(task['iot'], 0) + task['remain']

        return activated_queues

    def _process_edge_comp_in_concurrent(self, edge_time_remain, edge_index):
        activated_queues = self._activated_comp_task(edge_index)
        for iot_index in activated_queues.keys():
            edge_time_remain -= self.env.edge_server_cache[edge_index].dep_container(container_index=iot_index)

        while edge_time_remain > 0 and len(activated_queues) > 0:
            comp_cap_edge = self.env.comp_cap_edge[edge_index] / len(activated_queues)
            min_remain_index = min(activated_queues, key=activated_queues.get)
            min_remain_time = activated_queues[min_remain_index] * self.env.comp_density[min_remain_index] / comp_cap_edge

            for iot_index, remain_task in activated_queues.items():
                remain_time = min_remain_time.copy()

                while remain_task > 0 and remain_time > 0:
                    get_task = next((task for task in self.env.Queue_edge_comp[edge_index] if task['iot'] == iot_index), None)
                    iot_comp_density = self.env.comp_density[iot_index]
                    if remain_time * comp_cap_edge / iot_comp_density >= get_task['remain']:
                        remain_task -= get_task['remain']
                        remain_time -= get_task['remain'] * iot_comp_density / comp_cap_edge
                        self.env.process_delay[get_task['time'], iot_index] = \
                            self.env.time_count * self.env.duration + (self.env.duration - edge_time_remain) \
                            + (min_remain_time - remain_time) - get_task['time'] * self.env.duration
                        try:
                            self.env.Queue_edge_comp[edge_index].remove(get_task)
                        except ValueError:
                            print(f"Element {get_task} not found in deque.")
                    else:
                        get_task['remain'] -= remain_time * comp_cap_edge / iot_comp_density
                        remain_time = 0

            edge_time_remain -= min_remain_time
            activated_queues = self._activated_comp_task(edge_index)

    def update_info_for_opt(self):
        # INFO FOT OPTIMIZATION
        self.queue_iot_comp_remain: np.ndarray = np.zeros(self.env.n_iot)
        self.queue_iot_tran_remain: np.ndarray = np.zeros([self.env.n_iot, self.env.n_edge])
        self.queue_fog_comp_remain: np.ndarray = np.zeros([self.env.n_iot, self.env.n_edge])

        for iot_index in range(self.env.n_iot):
            for iot_comp_task in self.env.Queue_iot_comp[iot_index]:
                self.queue_iot_comp_remain[iot_index] += iot_comp_task['remain']

            for edge_index in range(self.env.n_edge):
                for iot_tran_task in self.env.Queue_iot_tran[iot_index][edge_index]:
                    self.queue_iot_tran_remain[iot_index, edge_index] += iot_tran_task['remain']

        for edge_index in range(self.env.n_edge):
            for edge_comp_task in self.env.Queue_edge_comp[edge_index]:
                self.queue_fog_comp_remain[edge_comp_task['iot'], edge_index] += edge_comp_task['remain']

        self.y_pre: np.ndarray = np.zeros([self.env.n_iot, self.env.n_edge])
        self.z_pre: np.ndarray = np.zeros([self.env.n_iot, self.env.n_edge])

        for iot_index in range(self.env.n_iot):
            for edge_index in range(self.env.n_edge):
                if self.env.edge_server_cache[edge_index].has_container(container_index=iot_index):
                    self.y_pre[iot_index, edge_index] = 1
                else:
                    self.y_pre[iot_index, edge_index] = 0
                if self.env.edge_server_cache[edge_index].has_image(image_index=iot_index):
                    self.z_pre[iot_index, edge_index] = 1
                else:
                    self.z_pre[iot_index, edge_index] = 0
