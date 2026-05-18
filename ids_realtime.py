import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
ALERT_LOG  = Path("ids_alerts.jsonl")

# ── Protocol numbers ──────────────────────────────────────────────────────────
PROTO_MAP = {1: "ICMP", 6: "TCP", 17: "UDP"}
COMMON_PORTS = {80, 443, 22, 53, 25, 110, 143, 8080, 3306}

# ANSI colours
RED  = "\033[91m"
YLW  = "\033[93m"
GRN  = "\033[92m"
RST  = "\033[0m"
BOLD = "\033[1m"


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(pkt) -> np.ndarray | None:
    """
    Extract a fixed-length feature vector from a Scapy packet.
    Returns None if the packet cannot be parsed.
    """
    try:
        from scapy.layers.inet import IP, TCP, UDP, ICMP

        if not pkt.haslayer(IP):
            return None

        ip   = pkt[IP]
        proto_num = ip.proto                          
        proto_enc = {1: 0, 6: 2, 17: 1}.get(proto_num, 0)

        pkt_len = len(pkt)

        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            pkt_type = 1  # Data-ish
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            pkt_type = 1
        else:
            src_port = 0
            dst_port = 0
            pkt_type = 0  

        
        anomaly = 0.0
        if dst_port not in COMMON_PORTS and dst_port != 0:
            anomaly += 30
        if pkt_len > 1200:
            anomaly += 25
        if src_port > 60000:
            anomaly += 20
        anomaly = min(anomaly + np.random.uniform(0, 10), 100.0)

        
        if dst_port == 53 or src_port == 53:
            traffic_enc = 0
        elif dst_port in {20, 21}:
            traffic_enc = 1
        else:
            traffic_enc = 2

        
        severity_enc = 0 if anomaly < 33 else (1 if anomaly < 66 else 2)

        
        last_oct = int(str(ip.src).split(".")[-1])
        seg_enc  = last_oct % 3

        
        action_enc = 1  

        port_ratio   = src_port / (dst_port + 1)
        len_per_port = pkt_len / (dst_port + 1)

        feature_vector = np.array([
            src_port, dst_port, pkt_len, anomaly,
            proto_enc, traffic_enc, pkt_type, severity_enc,
            seg_enc, action_enc,
            port_ratio, len_per_port,
        ], dtype=np.float32)

        return feature_vector, {
            "src_ip":    str(ip.src),
            "dst_ip":    str(ip.dst),
            "protocol":  PROTO_MAP.get(proto_num, str(proto_num)),
            "src_port":  src_port,
            "dst_port":  dst_port,
            "pkt_len":   pkt_len,
            "anomaly":   round(anomaly, 2),
        }

    except Exception:
        return None


class IDSEngine:
    def __init__(self, mode: str = "ensemble"):
        self.mode   = mode
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.xgb    = joblib.load(MODELS_DIR / "xgboost.pkl")
        self.iso    = joblib.load(MODELS_DIR / "isolation_forest.pkl")

        from utils.autoencoder import NumpyAutoencoder
        self.ae       = NumpyAutoencoder.load(MODELS_DIR / "autoencoder.npz")
        self.ae_thresh = 0.07  

        self.counts  = {"total": 0, "attack": 0, "normal": 0, "blocked": 0}
        self.start_t = time.time()

    def predict(self, fvec: np.ndarray) -> dict:
        X = self.scaler.transform(fvec.reshape(1, -1))

        xgb_p    = int(self.xgb.predict(X)[0])
        xgb_prob = float(self.xgb.predict_proba(X)[0, 1])

        iso_raw  = self.iso.predict(X)[0]
        iso_p    = 1 if iso_raw == -1 else 0

        ae_err   = float(self.ae.reconstruction_error(X)[0])
        ae_p     = 1 if ae_err > self.ae_thresh else 0

        if self.mode == "xgboost":
            verdict = xgb_p
        elif self.mode == "isolation_forest":
            verdict = iso_p
        elif self.mode == "autoencoder":
            verdict = ae_p
        else: 
            verdict = int((xgb_p + iso_p + ae_p) >= 2)

        return {
            "verdict":    verdict,
            "xgb_prob":   round(xgb_prob, 4),
            "iso_pred":   iso_p,
            "ae_error":   round(ae_err, 5),
            "ae_pred":    ae_p,
        }

    def handle_packet(self, pkt):
        result = extract_features(pkt)
        if result is None:
            return
        fvec, meta = result
        self.counts["total"] += 1

        pred = self.predict(fvec)
        ts   = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if pred["verdict"] == 1:
            self.counts["attack"] += 1
            action = "BLOCK" if pred["xgb_prob"] > 0.85 else "ALERT"
            if action == "BLOCK":
                self.counts["blocked"] += 1
            colour = RED
        else:
            self.counts["normal"] += 1
            action = "ALLOW"
            colour = GRN

  
        print(
            f"{colour}[{ts}] {action:5s}{RST}  "
            f"{meta['src_ip']:>15}:{meta['src_port']:<5}  →  "
            f"{meta['dst_ip']:>15}:{meta['dst_port']:<5}  "
            f"{meta['protocol']:4s}  len={meta['pkt_len']:4d}  "
            f"score={meta['anomaly']:5.1f}  xgb={pred['xgb_prob']:.3f}  ae={pred['ae_error']:.4f}"
        )

        
        event = {
            "timestamp": datetime.now().isoformat(),
            "action":    action,
            **meta,
            **pred,
        }
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")

       
        if self.counts["total"] % 100 == 0:
            elapsed  = time.time() - self.start_t
            pps      = self.counts["total"] / elapsed
            atk_pct  = 100 * self.counts["attack"] / max(1, self.counts["total"])
            print(
                f"\n{BOLD}── Stats @ {self.counts['total']} packets ──{RST}  "
                f"rate={pps:.1f} pkt/s  attacks={atk_pct:.1f}%  "
                f"blocked={self.counts['blocked']}\n"
            )



def run_demo(engine: IDSEngine, n: int = 300, delay: float = 0.15):
    """Synthetic packet generator for environments without a NIC."""
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.packet import Raw
    rng = np.random.default_rng(0)

    print(f"\n{BOLD}[DEMO] Generating {n} synthetic packets…{RST}\n")
    for _ in range(n):
        proto = rng.choice(["tcp", "udp", "icmp"])
        src   = f"{rng.integers(1,254)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}"
        dst   = f"192.168.{rng.integers(0,5)}.{rng.integers(1,254)}"
        if proto == "tcp":
            pkt = IP(src=src, dst=dst) / TCP(sport=int(rng.integers(1024,65535)), dport=int(rng.choice([80,443,22,8080,31337,4444])))
        elif proto == "udp":
            pkt = IP(src=src, dst=dst) / UDP(sport=int(rng.integers(1024,65535)), dport=int(rng.choice([53,123,5353,4500])))
        else:
            pkt = IP(src=src, dst=dst) / ICMP()
        pkt = pkt / Raw(load=bytes(rng.integers(0,256,rng.integers(0,1400)).tolist()))
        engine.handle_packet(pkt)
        time.sleep(delay)




def main():
    ap = argparse.ArgumentParser(description="AI-IDS real-time packet analyser")
    ap.add_argument("--iface", default=None, help="Network interface (e.g. eth0)")
    ap.add_argument("--mode",  default="ensemble",
                    choices=["xgboost","isolation_forest","autoencoder","ensemble"])
    ap.add_argument("--demo",  action="store_true",
                    help="Run with synthetic packets (no root / NIC required)")
    ap.add_argument("--count", type=int, default=0,
                    help="Stop after N packets (0 = run forever)")
    args = ap.parse_args()

    print(f"\n{BOLD}╔══════════════════════════════════════════════╗{RST}")
    print(f"{BOLD}║   AI Intrusion Detection System — Live IDS   ║{RST}")
    print(f"{BOLD}╚══════════════════════════════════════════════╝{RST}")
    print(f"  Mode: {args.mode}  |  Interface: {args.iface or 'auto'}  |  Demo: {args.demo}\n")

    engine = IDSEngine(mode=args.mode)

    if args.demo:
        run_demo(engine, n=args.count if args.count else 300)
        return

    try:
        from scapy.all import sniff
    except ImportError:
        print("Scapy not installed. Run:  pip install scapy")
        sys.exit(1)

    kwargs = {"prn": engine.handle_packet, "store": False}
    if args.iface:
        kwargs["iface"] = args.iface
    if args.count:
        kwargs["count"] = args.count

    print("Sniffing — press Ctrl+C to stop\n")
    try:
        sniff(**kwargs)
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - engine.start_t
    print(f"\n{BOLD}── Final Summary ──{RST}")
    print(f"  Total packets : {engine.counts['total']}")
    print(f"  Attacks       : {engine.counts['attack']}")
    print(f"  Normal        : {engine.counts['normal']}")
    print(f"  Blocked       : {engine.counts['blocked']}")
    print(f"  Duration      : {elapsed:.1f}s")
    print(f"  Alert log     : {ALERT_LOG}")


if __name__ == "__main__":
    main()
