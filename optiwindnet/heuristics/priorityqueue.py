# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import itertools
from heapq import heappop, heappush

__all__ = ()


class PriorityQueue(list):
    def __init__(self):
        super().__init__(self)
        # self.entries = []
        self.tags = {}
        self.counter = itertools.count()

    def add(self, priority, tag, payload):
        """lowest priority pops first, payload cannot be None.
        an addition with an already existing tag will cancel the
        previous entry."""
        if payload is None:
            raise ValueError('payload cannot be None.')
        if tag in self.tags:
            self.cancel(tag)
        entry = [priority, next(self.counter), tag, payload]
        self.tags[tag] = entry
        heappush(self, entry)

    def strip(self):
        """removes all cancelled entries from the queue top"""
        while self:
            if self[0][-1] is None:
                heappop(self)
            else:
                break

    def cancel(self, tag):
        entry = self.tags.pop(tag)
        entry[-1] = None
        self.strip()

    def top(self):
        "returns the payload with lowest priority"
        priority, count, tag, payload = heappop(self)
        del self.tags[tag]
        self.strip()
        return tag, payload
