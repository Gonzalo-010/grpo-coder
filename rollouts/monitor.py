"""Monitor de GPU y del sistema en un hilo (cada monitor.interval_s), a gpu.jsonl.

GPU: NVML si está (memoria, uso, temperatura, potencia, tráfico PCIe); si no, torch.cuda.mem_get_info. Campos no
soportados (p. ej. potencia o PCIe en WSL2) -> None.
Sistema (Linux /proc): RAM disponible, swap usada, RSS de este proceso, CPU del sistema (%) y procesos hijos zombie.
Sirve para diagnosticar steps lentos: paginación de VRAM (uso alto + potencia baja + PCIe alto), swapping de RAM,
contención de CPU o procesos del sandbox que no se recogen.
"""
import json, os, threading, time

try:
    import pynvml
    pynvml.nvmlInit()
    _H = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml, _H = None, None


def _try(f, *a):
    try:
        return f(*a)
    except Exception:
        return None


def _meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    out[k] = int(v.split()[0]) >> 10  # MB
    except (OSError, ValueError):
        pass
    return out


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) >> 10
    except (OSError, ValueError):
        pass
    return None


def _cpu_times():
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0)  # total, idle + iowait
    except (OSError, ValueError, IndexError):
        return None


def zombies(pid=None):
    """Hijos de `pid` (por defecto este proceso) en estado Z: subprocesos que nadie recogió."""
    pid, n = pid or os.getpid(), 0
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    for p in names:
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                rest = f.read().rsplit(")", 1)[1].split()
            n += rest[0] == "Z" and int(rest[1]) == pid
        except (OSError, IndexError, ValueError):
            continue
    return n


_LAST_CPU = {"v": None}


def snapshot():
    s = {"t": round(time.time(), 1)}
    if pynvml:
        m = _try(pynvml.nvmlDeviceGetMemoryInfo, _H)
        if m:
            s.update(used_mb=m.used >> 20, free_mb=m.free >> 20, total_mb=m.total >> 20)
        u = _try(pynvml.nvmlDeviceGetUtilizationRates, _H)
        s["util"] = u.gpu if u else None
        s["temp_c"] = _try(pynvml.nvmlDeviceGetTemperature, _H, pynvml.NVML_TEMPERATURE_GPU)
        p = _try(pynvml.nvmlDeviceGetPowerUsage, _H)
        s["power_w"] = round(p / 1000, 1) if p is not None else None
        for key, kind in (("pcie_rx_mbs", "NVML_PCIE_UTIL_RX_BYTES"), ("pcie_tx_mbs", "NVML_PCIE_UTIL_TX_BYTES")):
            v = _try(pynvml.nvmlDeviceGetPcieThroughput, _H, getattr(pynvml, kind, 0))
            s[key] = round(v / 1024, 1) if v is not None else None  # KB/s -> MB/s
    if "used_mb" not in s:
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            s.update(used_mb=(total - free) >> 20, free_mb=free >> 20, total_mb=total >> 20)
        except Exception:
            pass
    mi = _meminfo()
    if mi:
        s["ram_avail_mb"] = mi.get("MemAvailable")
        if "SwapTotal" in mi:
            s["swap_used_mb"] = mi["SwapTotal"] - mi.get("SwapFree", mi["SwapTotal"])
    s["rss_mb"] = _rss_mb()
    ct = _cpu_times()
    if ct and _LAST_CPU["v"]:
        dt, di = ct[0] - _LAST_CPU["v"][0], ct[1] - _LAST_CPU["v"][1]
        s["cpu_pct"] = round(100.0 * (dt - di) / dt, 1) if dt > 0 else None
    _LAST_CPU["v"] = ct
    return s


def window(rows, t0, t1):
    """Muestras del monitor con t en [t0, t1]."""
    return [r for r in rows if t0 <= r.get("t", 0) <= t1]


def summarize(rows):
    """Resumen de una ventana: máximos/medias de lo que importa para un step."""
    col = lambda k: [r[k] for r in rows if r.get(k) is not None]
    mx = lambda k: max(col(k), default=None)
    mean = lambda k: round(sum(col(k)) / len(col(k)), 1) if col(k) else None
    return dict(samples=len(rows), vram_used_max_mb=mx("used_mb"), util_mean=mean("util"), power_mean_w=mean("power_w"),
                temp_max_c=mx("temp_c"), pcie_rx_max_mbs=mx("pcie_rx_mbs"), pcie_tx_max_mbs=mx("pcie_tx_mbs"),
                cpu_mean_pct=mean("cpu_pct"), ram_avail_min_mb=min(col("ram_avail_mb"), default=None),
                swap_used_max_mb=mx("swap_used_mb"), rss_max_mb=mx("rss_mb"))


class GpuMonitor:
    def __init__(self, interval=2.0, log_path=None, keep=4096):
        self.interval, self.log_path, self.rows, self.keep = interval, log_path, [], keep
        self.n = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._agg = {}

    def _loop(self):
        f = open(self.log_path, "a") if self.log_path else None
        while True:
            s = snapshot()
            if self.n % 15 == 0:  # cada ~30 s: escanear /proc entero es caro para hacerlo en cada muestra
                s["zombies"] = zombies()
            self.n += 1
            self.rows.append(s)
            self._fold(s)
            if len(self.rows) > self.keep:  # memoria acotada en runs largos; el agregado global sigue completo
                del self.rows[:len(self.rows) - self.keep]
            if f:
                f.write(json.dumps(s) + "\n")
                f.flush()
            if self._stop.wait(self.interval):
                break
        if f:
            f.close()

    def _fold(self, s):
        a = self._agg
        for k, fn in (("used_mb", max), ("total_mb", max), ("temp_c", max), ("power_w", max)):
            if s.get(k) is not None:
                a[k] = fn(a.get(k, s[k]), s[k])
        if s.get("free_mb") is not None:
            a["free_mb"] = min(a.get("free_mb", s["free_mb"]), s["free_mb"])
        if s.get("util") is not None:
            a["util_sum"], a["util_n"] = a.get("util_sum", 0) + s["util"], a.get("util_n", 0) + 1

    def start(self):
        self._th.start()
        return self

    def recent(self, t0, t1=None):
        return window(list(self.rows), t0, t1 or time.time() + 1)

    def stop(self):
        self._stop.set()
        self._th.join(timeout=5)
        a = self._agg
        return {"samples": self.n, "vram_used_peak_mb": a.get("used_mb"), "vram_free_min_mb": a.get("free_mb"),
                "total_mb": a.get("total_mb"),
                "util_mean": round(a["util_sum"] / a["util_n"], 1) if a.get("util_n") else None,
                "temp_max_c": a.get("temp_c"), "power_max_w": a.get("power_w")}
