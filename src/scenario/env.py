import numpy as np
from collections import deque

from src.scenario.cache import EdgeServerCache


class Env:
    def __init__(self, num_iot, num_edge, num_time, duration, task_arrive_prob, max_bit_arrive, min_bit_arrive,
                 comp_cap_iot, comp_cap_edge, tran_cap_iot, comp_density, container_size, container_startup,
                 container_delete, image_size, image_delete, edge_container_cache_limit, edge_image_cache_limit,
                 edge_download_speed):
        # INPUT DATA
        self.n_iot = num_iot
        self.n_edge = num_edge
        self.n_time = num_time
        self.duration = duration

        # BIT_ARRIVE_SET
        self.task_arrive_prob = task_arrive_prob
        self.max_bit_arrive = max_bit_arrive
        self.min_bit_arrive = min_bit_arrive

        self.bit_arrive = np.zeros([self.n_time, self.n_iot])

        self.comp_cap_iot = comp_cap_iot * np.ones(self.n_iot)
        self.comp_cap_edge = comp_cap_edge * np.ones([self.n_edge])

        # UNIQUE TRANS CAP
        uniform_values: np.ndarray = np.linspace(tran_cap_iot * 0.99, tran_cap_iot * 1.01, self.n_iot * self.n_edge)
        np.random.shuffle(uniform_values)
        self.tran_cap_iot = uniform_values.reshape((self.n_iot, self.n_edge))

        # SAME TRANS CAP
        # self.tran_cap_iot = tran_cap_iot * np.ones((self.n_iot, self.n_edge))

        self.comp_density = comp_density * np.ones([self.n_iot])

        # ACTIONS (0 for LOCAL, 1-N for EDGE)。
        self.n_actions = 1 + num_edge

        # CONTAINERS
        self.container_size = container_size * np.ones([self.n_iot])
        self.container_startup = container_startup * np.ones([self.n_iot])
        self.container_delete = container_delete * np.ones([self.n_iot])
        self.container_set = [
            {'size': self.container_size[container_index],
             'startup_time': self.container_startup[container_index],
             'delete_time': self.container_delete[container_index]}
            for container_index in range(self.n_iot)
        ]
        # IMAGES
        self.image_size = image_size * np.ones([self.n_iot])
        self.image_delete = image_delete * np.ones([self.n_iot])
        self.image_set = [
            {'size': self.image_size[image_index],
             'delete_time': self.image_delete[image_index]}
            for image_index in range(self.n_iot)
        ]
        # CACHING
        self.edge_container_cache_limit = edge_container_cache_limit * np.ones([self.n_edge])
        self.edge_image_cache_limit = edge_image_cache_limit * np.ones([self.n_edge])
        self.edge_download_speed = edge_download_speed * np.ones([self.n_edge])
        self.edge_server_cache = [
            EdgeServerCache(
                container_set=self.container_set,
                image_set=self.image_set,
                container_cache_limit=self.edge_container_cache_limit[edge_index],
                image_cache_limit=self.edge_image_cache_limit[edge_index],
                download_speed=self.edge_download_speed[edge_index]
            )
            for edge_index in range(self.n_edge)
        ]

        # CACHE SOME IMAGES AND CONTAINERS
        for iot_index in range(self.n_iot):
            dep_prob = 0.5
            if np.random.uniform(0, 1) < dep_prob:
                edge_index = np.random.choice(range(self.n_edge))
                self.edge_server_cache[edge_index].download_image(image_index=iot_index)
                self.edge_server_cache[edge_index].dep_container(container_index=iot_index)

        # TIME COUNT
        self.time_count = int(0)

        self.Queue_iot_comp = list()
        self.Queue_iot_tran = list()
        self.Queue_edge_wait = list()
        self.Queue_edge_comp = list()

        for iot in range(self.n_iot):
            self.Queue_iot_comp.append(deque())
            self.Queue_iot_tran.append(list())
            self.Queue_edge_wait.append(list())
            for edge in range(self.n_edge):
                self.Queue_iot_tran[iot].append(deque())
                self.Queue_edge_wait[iot].append(deque())
        for edge in range(self.n_edge):
            self.Queue_edge_comp.append(deque())

        # TASK DELAY
        self.process_delay = np.zeros([self.n_time, self.n_iot])
        self.process_delay_trans = np.zeros([self.n_time, self.n_iot])
