import random


class Container:
    def __init__(self, index, size, startup_time, delete_time):
        self.index = index
        self.size = size
        self.startup_time = startup_time
        self.delete_time = delete_time


class Image:
    def __init__(self, index, size, delete_time):
        self.index = index
        self.size = size
        self.delete_time = delete_time


class EdgeServerCache:
    def __init__(self, container_set, image_set, container_cache_limit, image_cache_limit, download_speed):
        self.container_set = [
            Container(index=index,
                      size=container['size'],
                      startup_time=container['startup_time'],
                      delete_time=container['delete_time'])
            for index, container in enumerate(container_set)
        ]
        self.image_set = [
            Image(index=index,
                  size=image['size'],
                  delete_time=image['delete_time'])
            for index, image in enumerate(image_set)
        ]
        self.container_cache = {}
        self.image_cache = {}
        self.container_cache_limit = container_cache_limit
        self.image_cache_limit = image_cache_limit
        self.current_container_cache_usage = 0
        self.current_image_cache_usage = 0
        self.download_speed = download_speed

    def cache_container(self, container_index):
        if self.current_container_cache_usage + self.container_set[container_index].size <= self.container_cache_limit:
            self.container_cache[container_index] = self.container_set[container_index]
            self.current_container_cache_usage += self.container_set[container_index].size
            return True
        return False

    def cache_image(self, image_index):
        if self.current_image_cache_usage + self.image_set[image_index].size <= self.image_cache_limit:
            self.image_cache[image_index] = self.image_set[image_index]
            self.current_image_cache_usage += self.image_set[image_index].size
            return True
        return False

    def has_container(self, container_index):
        return container_index in self.container_cache

    def has_image(self, image_index):
        return image_index in self.image_cache

    def download_image(self, image_index):
        download_time = self.image_set[image_index].size / self.download_speed
        delete_time = 0
        flag = self.cache_image(image_index)

        while not flag:
            # delete some image cache
            random_image_index = random.choice(list(self.image_cache.keys()))
            delete_time += self.del_image(image_index=random_image_index)

            flag = self.cache_image(image_index)

        return download_time + delete_time

    def dep_container(self, container_index):
        deploy_time = 0

        if self.has_container(container_index):
            return 0

        if not self.has_image(image_index=container_index):
            deploy_time += self.download_image(container_index)

        if self.has_image(container_index):
            deploy_time += self.container_set[container_index].startup_time
            delete_time = 0
            flag = self.cache_container(container_index)

            while not flag:
                # delete some container cache
                random_container_index = random.choice(list(self.container_cache.keys()))
                delete_time += self.del_container(container_index=random_container_index)

                flag = self.cache_container(container_index)

            return deploy_time + delete_time

        else:
            print(f"Failed to deploy container {container_index}. No such corresponding image.")
            return deploy_time

    def del_image(self, image_index):
        if self.has_image(image_index):
            image = self.image_cache.pop(image_index)
            self.current_image_cache_usage -= image.size
            return image.delete_time
        else:
            print(f"Failed to delete image {image_index}. No such image.")
            return 0

    def del_container(self, container_index):
        if self.has_container(container_index):
            container = self.container_cache.pop(container_index)
            self.current_container_cache_usage -= container.size
            return container.delete_time
        else:
            print(f"Failed to delete container {container_index}. No such container.")
            return 0
