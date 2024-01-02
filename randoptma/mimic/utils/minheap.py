""" Class for a minimum priority sorting queue """

# Author: Mohamed Abouelsaadat
# License: MIT

import heapq


class MinHeapQueue(object):
    def __init__(self) -> None:
        self.queue = []

    def __len__(self) -> int:
        return len(self.queue)

    def push(self, item: tuple) -> None:
        heapq.heappush(self.queue, item)

    def top(self) -> tuple:
        return self.queue[0] if len(self.queue) > 0 else None

    def pop(self) -> tuple:
        return heapq.heappop(self.queue) if len(self.queue) > 0 else None
