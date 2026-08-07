"""Validate that the buffered TensorBoard writer produces identical scalar
streams to the stock writer (content, tags, order), and report the speedup.

Usage: python -m tools.check_tb_writer [log_root]
"""

import shutil
import sys
import tempfile
import time

from torch.utils.tensorboard import SummaryWriter

from utils.tb_writer import create_summary_writer

TAGS = [f"train/metric_{i}" for i in range(20)] + [
    f"running_stat/{n}" for n in ("it_s", "rows", "lr")
]
STEPS = 30


def fill(writer):
    for step in range(STEPS):
        for j, tag in enumerate(TAGS):
            writer.add_scalar(tag, step * 0.5 + j, step)
    writer.close()


def read_all(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(logdir)
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="tbcheck")
    refs, news = {}, {}
    for name, factory in (("stock", SummaryWriter), ("buffered", create_summary_writer)):
        d = f"{root}/{name}"
        shutil.rmtree(d, ignore_errors=True)
        t0 = time.perf_counter()
        fill(factory(d))
        dt = time.perf_counter() - t0
        (refs if name == "stock" else news).update(read_all(d))
        print(f"{name:9s}: {len(TAGS) * STEPS} scalars in {dt * 1000:.1f} ms")
        shutil.rmtree(d, ignore_errors=True)
    if refs != news:
        missing = set(refs) ^ set(news)
        bad = [t for t in refs if refs[t] != news.get(t)]
        print(f"MISMATCH: tags {missing or bad}")
        return 1
    print("OK: identical scalar streams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
