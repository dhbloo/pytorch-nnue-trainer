"""Buffered TensorBoard event writer for latency-bound filesystems.

On network/fs-proxy filesystems (notably WSL drvfs mounts), every small
synchronous write costs a fixed round trip of ~0.3 ms — and the stock
``tensorboard`` pipeline frames each scalar event into its own tiny write
syscalls.  Because the async writer thread's byte queue has ``maxsize=10``,
a burst of ~150 scalars at a log boundary ends up blocking the training
thread for hundreds of milliseconds per log interval (measured: ~300 ms per
45 scalars on ``/mnt/...`` vs ~6 ms on a native filesystem; ~900 ms with the
data-pipeline observability scalars enabled).

The fix preserves the on-disk TFRecord framing byte-for-byte: a buffering
shim sits *below* ``RecordWriter`` (which keeps framing each record exactly
as before) and only emits the accumulated bytes to the real file when the
capacity threshold or a flush is reached.  Readers (EventAccumulator,
TensorBoard) see identical files, just produced with far fewer syscalls.

``create_summary_writer`` swaps the shim into a stock
``torch.utils.tensorboard.SummaryWriter``.  If the tensorboard internals do
not match the attributes this graft expects (library upgrade), it logs a
warning and returns the unmodified writer — logging must never take down a
training run.
"""

import logging

from torch.utils.tensorboard import SummaryWriter

_DEFAULT_BUFFER_CAPACITY = 256 * 1024


class _BufferedFileLike:
    """File-like shim aggregating small writes; byte stream identical to inner."""

    def __init__(self, inner, capacity=_DEFAULT_BUFFER_CAPACITY):
        self._inner = inner
        self._buf = bytearray()
        self._capacity = capacity

    def write(self, data):
        self._buf += data
        if len(self._buf) >= self._capacity:
            self._drain()
        return len(data)

    def _drain(self):
        if self._buf:
            self._inner.write(bytes(self._buf))
            self._buf.clear()

    def flush(self):
        self._drain()
        self._inner.flush()

    def close(self):
        try:
            self._drain()
        finally:
            self._inner.close()

    @property
    def closed(self):
        return self._inner.closed

    def __getattr__(self, name):
        return getattr(self._inner, name)


def buffer_tensorboard_writer(writer, capacity=_DEFAULT_BUFFER_CAPACITY):
    """Install write buffering on an open SummaryWriter.

    Returns *writer* (possibly the same object, unchanged) so callers can use
    it in an expression.  Never raises.
    """
    try:
        event_file_writer = writer.file_writer.event_writer
        record_writer = event_file_writer._async_writer._writer
        record_writer._writer = _BufferedFileLike(record_writer._writer, capacity)
    except AttributeError:
        logging.warning(
            "Failed to install buffered TensorBoard writer; tensorboard "
            "internals changed. Logging will use the stock writer."
        )
    return writer


def create_summary_writer(log_dir):
    """Create a SummaryWriter whose event file writes are buffered."""
    return buffer_tensorboard_writer(SummaryWriter(log_dir))
