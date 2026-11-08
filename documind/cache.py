"""A tiny bounded LRU cache (Day 18).

Standard library only — :class:`collections.OrderedDict` does the
heavy lifting. The cache backs the query-result cache in
:mod:`documind.index`: repeated searches for the same query return
the stored results instead of re-scoring the whole index, which is
what makes the web UI's question history and the API's repeat
queries cheap.

A ``maxsize`` of 0 disables the cache entirely (nothing is stored,
every lookup is a miss); that is how ``query_cache_size: 0`` in the
config turns caching off. ``stats()`` reports hits, misses and
evictions so the cost/benefit of the cache is measurable rather than
assumed — see :mod:`documind.benchmark` and ``cli.py bench``.
"""
from collections import OrderedDict


class LRUCache:
    """A least-recently-used cache with a fixed maximum size.

    ``maxsize`` must be a non-negative integer. When the cache is
    full, inserting a new key evicts the least recently used entry;
    updating an existing key refreshes its recency without evicting
    anything. ``get`` returns ``None`` for a miss (cached values are
    never ``None`` here — search results are always lists), and every
    lookup updates the hit/miss counters.
    """

    def __init__(self, maxsize=128):
        if isinstance(maxsize, bool) or not isinstance(maxsize, int):
            raise ValueError("maxsize must be an integer, got %r"
                             % (maxsize,))
        if maxsize < 0:
            raise ValueError("maxsize must be at least 0, got %d"
                             % maxsize)
        self.maxsize = maxsize
        self._entries = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key, default=None):
        """Return the cached value for ``key``, or ``default``.

        A hit moves the entry to the most-recently-used end; a miss
        leaves the cache untouched.
        """
        try:
            value = self._entries.pop(key)
        except KeyError:
            self.misses += 1
            return default
        self._entries[key] = value
        self.hits += 1
        return value

    def put(self, key, value):
        """Store ``value`` under ``key``, evicting the LRU entry if
        the cache is full. A no-op when ``maxsize`` is 0 (disabled)."""
        if self.maxsize == 0:
            return
        if key in self._entries:
            del self._entries[key]
        elif len(self._entries) >= self.maxsize:
            self._entries.popitem(last=False)
            self.evictions += 1
        self._entries[key] = value

    def clear(self):
        """Drop every entry (counters are kept: they describe the
        cache's lifetime, not its current contents)."""
        self._entries.clear()

    def __contains__(self, key):
        return key in self._entries

    def __len__(self):
        return len(self._entries)

    def stats(self):
        """``{"size", "maxsize", "hits", "misses", "evictions"}``."""
        return {"size": len(self._entries),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions}
