""" Class for a minimum priority sorting queue """

# Author: Mohamed Abouelsaadat
# License: MIT

import heapq


class MinHeapQueue(object):
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def push(self, item):
        heapq.heappush(self.queue, item)

    def top(self):
        return self.queue[0] if len(self.queue) > 0 else None

    def pop(self):
        return heapq.heappop(self.queue) if len(self.queue) > 0 else None
