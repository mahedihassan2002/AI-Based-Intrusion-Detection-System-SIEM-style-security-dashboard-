# 🛡️ AI-Based Intrusion Detection System (IDS)

> Detects malicious network traffic using **Isolation Forest**, **XGBoost**, and a **NumPy Autoencoder** — with a real-time Scapy packet sniffer and a SIEM-style Streamlit dashboard.

---

## 📁 Project Structure

```
ids_project/
├── ids_main.py           ← Training pipeline (all 3 models)
├── ids_realtime.py       ← Real-time packet capture (Scapy)
├── utils/
│   ├── preprocess.py     ← Feature engineering + synthetic normals
│   ├── autoencoder.py    ← Pure NumPy autoencoder (no TF/PyTorch)
│   └── visualise.py      ← Builds dashboard_data.json
├── dashboard/
│   └── app.py            ← Streamlit SIEM dashboard
├── data/
│   └── cybersecurity_attacks.csv   ← Kaggle dataset
├── models/               ← Saved model artefacts (after training)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the dataset

```bash
cp cybersecurity_attacks.csv data/
```

### 3. Train all models

```bash
python ids_main.py
```

Output:
```
== Isolation Forest ==   precision 0.86  recall 0.50  AUC 0.51
== XGBoost ==            precision 1.00  recall 1.00  AUC 1.00
== Autoencoder ==        precision 0.91  recall 0.88  AUC 0.93
== Ensemble ==           majority vote of the above
```

### 4. Launch the SIEM dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Real-time packet capture

```bash
# Demo mode (no root / NIC required)
python ids_realtime.py --demo

# Live capture on a real interface (requires root)
sudo python ids_realtime.py --iface eth0 --mode ensemble
```

---

## 🧠 Models

| Model | Type | Technique | Strength |
|-------|------|-----------|----------|
| **Isolation Forest** | Unsupervised | Random partitioning | No labels needed; fast |
| **XGBoost** | Supervised | Gradient boosting | Near-perfect accuracy |
| **Autoencoder** | Unsupervised | Reconstruction error | Novel/zero-day detection |
| **Ensemble** | Hybrid | Majority vote (2/3) | Robust; best of all worlds |

---

## 🔢 Features Used

| Feature | Description |
|---------|-------------|
| Source Port | Originating port number |
| Destination Port | Target service port |
| Packet Length | Byte size of packet |
| Anomaly Score | Pre-computed threat score (0–100) |
| Protocol | TCP / UDP / ICMP |
| Traffic Type | HTTP / DNS / FTP |
| Packet Type | Data / Control |
| Severity Level | Low / Medium / High |
| Network Segment | Segment A / B / C |
| Action Taken | Blocked / Logged / Ignored |
| port_ratio | src_port / dst_port (derived) |
| len_per_port | packet_len / dst_port (derived) |

---

## 📊 Dataset

- **Source:** Kaggle — Cybersecurity Attack Dataset
- **Records:** 40,000 network events
- **Attack types:** DDoS (33.6%), Malware (33.3%), Intrusion (33.2%)
- **Note:** Dataset contains only attack traffic; `utils/preprocess.py` injects 8,000 synthetic normal packets for balanced training.

---

## 📡 Real-time Pipeline

```
NIC / Demo Generator
       │
       ▼
  Scapy Sniffer  (ids_realtime.py)
       │
       ▼
  Feature Extraction
  (port, length, protocol, anomaly heuristic, …)
       │
       ▼
  StandardScaler  (models/scaler.pkl)
       │
    ┌──┴────────────────┐
    │                   │
  XGBoost           IsoForest + AE
  (supervised)      (unsupervised)
    │                   │
    └──────┬────────────┘
           │ Ensemble majority vote
           ▼
      ALLOW / ALERT / BLOCK
           │
           ▼
     ids_alerts.jsonl  →  Streamlit dashboard
```

---

## 🖥️ Dashboard Panels

| Panel | Contents |
|-------|----------|
| **Overview** | KPIs, attack distribution donut, hourly timeline, protocol breakdown |
| **ML Models** | Accuracy/AUC table, confusion matrix, feature importance, AE error boxes |
| **Traffic** | Segment heatmap, action breakdown, anomaly score histogram |
| **Live Alerts** | Auto-refreshing feed from `ids_alerts.jsonl`, rolling anomaly chart |

---

## 🔧 Configuration

Edit constants at the top of each file:

| File | Variable | Default |
|------|----------|---------|
| `ids_main.py` | `DATA_PATH` | `data/cybersecurity_attacks.csv` |
| `ids_realtime.py` | `ALERT_LOG` | `ids_alerts.jsonl` |
| `ids_realtime.py` | `ae_thresh` | `0.07` |
| `utils/preprocess.py` | `n` (synthetic normals) | `8000` |

---

## 📜 License

MIT — for educational and research use.
