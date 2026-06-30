"""
ingest.py — Aviation Supply Chain: Data Ingestion Module
=========================================================
Phase 2 · Environment Setup & Data Ingestion

PURPOSE
-------
Downloads, caches, and loads all raw datasets used in the project.
- Kaggle datasets (Aerospace SC Performance, Global Aviation Safety, DataCo)
- BTS Airline On-Time Performance (from local CSV or BTS portal)
- OpenSky Network API (flight utilisation signals)
- supplychainpy (synthetic MRO inventory generator)

USAGE
-----
    from src.data.ingest import load_aerospace_sc, load_bts_ontime, generate_mro_synthetic

AUTHOR  : Aviation SC Analytics Project
VERSION : 1.0.0 (Phase 2)
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("aviation_sc.ingest")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]   # project root
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
SYNTH_DIR  = ROOT / "data" / "synthetic"

# Make sure directories exist
for d in [RAW_DIR, PROC_DIR, SYNTH_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. KAGGLE DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

def download_kaggle_dataset(dataset_slug: str, output_dir: Path = RAW_DIR) -> Path:
    """
    Download a Kaggle dataset using the Kaggle CLI.

    Parameters
    ----------
    dataset_slug : str
        Full Kaggle dataset identifier, e.g.
        "robertocarlost/aerospace-supply-chain-performance-and-forecasting"
    output_dir : Path
        Local folder where the ZIP will be unzipped.

    Returns
    -------
    Path
        Path to the folder containing the downloaded files.

    Requirements
    ------------
    - Kaggle API credentials must be saved at ~/.kaggle/kaggle.json
    - Install: pip install kaggle
    - Token: https://www.kaggle.com/account → Create New API Token
    """
    import subprocess
    folder_name = dataset_slug.split("/")[-1]
    dest = output_dir / folder_name

    if dest.exists() and any(dest.iterdir()):
        log.info(f"✅ Already downloaded: {folder_name} — skipping.")
        return dest

    log.info(f"⬇️  Downloading Kaggle dataset: {dataset_slug}")
    dest.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug,
         "--unzip", "-p", str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error(f"Kaggle download failed:\n{result.stderr}")
        raise RuntimeError(f"Failed to download {dataset_slug}. "
                           "Check kaggle.json credentials.")
    log.info(f"✅ Downloaded → {dest}")
    return dest


def load_aerospace_sc(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Kaggle 'Aerospace Supply Chain Performance & Forecasting' dataset.

    Columns include: supplier_id, part_category, lead_time_days,
    on_time_delivery_pct, defect_rate_ppm, order_quantity, unit_cost, etc.

    Returns
    -------
    pd.DataFrame
        Raw aerospace supply chain DataFrame.
    """
    slug   = "robertocarlost/aerospace-supply-chain-performance-and-forecasting"
    folder = RAW_DIR / slug.split("/")[-1]

    if not folder.exists() or force_download:
        download_kaggle_dataset(slug)

    # Find the CSV inside the folder
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder}. Re-run download.")

    df = pd.read_csv(csvs[0])
    log.info(f"✅ Aerospace SC dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_global_aviation_safety(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Kaggle 'Global Aviation Safety 1970–2024' dataset.

    Contains ~25,000 aviation incidents/accidents with maintenance flags.
    Used to correlate supply disruptions with safety outcomes.

    Returns
    -------
    pd.DataFrame
    """
    slug   = "khsamaha/aviation-accident-database-synopses"
    folder = RAW_DIR / slug.split("/")[-1]

    if not folder.exists() or force_download:
        download_kaggle_dataset(slug)

    csvs = list(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder}.")

    # Try UTF-8 first, fall back to latin-1 for aviation data
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(csvs[0], encoding=enc, low_memory=False)
            log.info(f"✅ Aviation Safety dataset loaded ({enc}): "
                     f"{df.shape[0]:,} rows × {df.shape[1]} cols")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode aviation safety CSV with any known encoding.")


def load_dataco_supply_chain(force_download: bool = False) -> pd.DataFrame:
    """
    Load the DataCo Smart Supply Chain dataset (~180K orders).

    Columns: Type, Days_for_shipping, Sales, Order_Status, Shipping_Mode,
    Latitude, Longitude, etc. Used for logistics cost benchmarking.

    Returns
    -------
    pd.DataFrame
    """
    slug   = "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
    folder = RAW_DIR / slug.split("/")[-1]

    if not folder.exists() or force_download:
        download_kaggle_dataset(slug)

    csvs = list(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder}.")

    df = pd.read_csv(csvs[0], encoding="latin-1")
    log.info(f"✅ DataCo SC dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BTS ON-TIME PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_bts_ontime(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load BTS Airline On-Time Performance data.

    If csv_path is None, looks for any CSV in data/raw/ whose name contains
    'ontime' or 'bts' (case-insensitive). Download manually from:
    https://www.transtats.bts.gov/DL_SelectFields.aspx

    Recommended columns to download from BTS:
        YEAR, MONTH, OP_CARRIER, ORIGIN, DEST, DEP_DELAY, ARR_DELAY,
        NAS_DELAY, LATE_AIRCRAFT_DELAY, CANCELLED, CANCELLATION_CODE

    Parameters
    ----------
    csv_path : str, optional
        Explicit path to the BTS CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if csv_path:
        path = Path(csv_path)
    else:
        # Auto-discover
        candidates = [
            f for f in RAW_DIR.glob("*.csv")
            if any(kw in f.name.lower() for kw in ["ontime", "bts", "airline"])
        ]
        if not candidates:
            log.warning("⚠️ No BTS CSV found. Returning empty DataFrame. "
                        "Download from https://www.transtats.bts.gov/")
            return pd.DataFrame()
        path = candidates[0]

    df = pd.read_csv(path, low_memory=False)
    log.info(f"✅ BTS On-Time data loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OPENSKY NETWORK API
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_opensky_flights(
    icao24: str,
    begin_ts: int,
    end_ts: int,
    username: str | None = None,
    password: str | None = None,
) -> pd.DataFrame:
    """
    Fetch historical flight records for a specific aircraft from OpenSky Network.

    Parameters
    ----------
    icao24 : str
        ICAO 24-bit hex address of the aircraft (e.g., "3c6757").
    begin_ts : int
        Start of time interval as Unix timestamp.
    end_ts : int
        End of time interval as Unix timestamp.
    username, password : str, optional
        OpenSky credentials for higher rate limits.

    Returns
    -------
    pd.DataFrame
        Columns: icao24, firstSeen, lastSeen, callsign, estDepartureAirport,
                 estArrivalAirport.

    Example
    -------
    >>> import time
    >>> end   = int(time.time())
    >>> begin = end - 30 * 86400   # last 30 days
    >>> df = fetch_opensky_flights("3c6757", begin, end)
    """
    url = "https://opensky-network.org/api/flights/aircraft"
    params = {"icao24": icao24, "begin": begin_ts, "end": end_ts}
    auth   = (username, password) if username else None

    try:
        resp = requests.get(url, params=params, auth=auth, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df   = pd.DataFrame(data)
        log.info(f"✅ OpenSky: {len(df)} flights fetched for {icao24}")
        return df
    except requests.RequestException as e:
        log.error(f"OpenSky API error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SYNTHETIC MRO INVENTORY DATA (supplychainpy)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_mro_synthetic(
    n_skus: int = 500,
    n_months: int = 24,
    random_seed: int = 42,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic MRO spare-parts inventory data.

    Since real MRO procurement data is often proprietary, we generate
    statistically realistic data using supplychainpy patterns + NumPy.

    Each row represents one SKU's monthly demand record. The dataset mimics:
    - Engine components (high-cost, low-volume, long lead time)
    - Airframe components (medium-cost, medium-volume)
    - Consumables (low-cost, high-volume, short lead time)
    - AOG-critical parts (highest priority, must-stock)

    Parameters
    ----------
    n_skus    : int  Number of unique spare-part SKUs to generate (default 500)
    n_months  : int  Number of months of history (default 24)
    random_seed : int  For reproducibility
    save      : bool  If True, saves CSV to data/synthetic/

    Returns
    -------
    pd.DataFrame
        Columns: sku_id, part_name, category, aircraft_type, unit_cost,
                 lead_time_days, monthly_demand, month_index, holding_cost_pct,
                 reorder_cost, safety_stock, eoq, abc_class, aog_critical
    """
    rng = np.random.default_rng(random_seed)

    # ── Part categories with realistic distributions ──
    categories = {
        "Engine Component":    {"n": int(n_skus * 0.25), "cost_range": (5_000, 250_000),
                                 "lead_range": (30, 120), "demand_mu": 3,  "demand_sig": 1.5},
        "Airframe Component":  {"n": int(n_skus * 0.30), "cost_range": (500,  15_000),
                                 "lead_range": (14,  60), "demand_mu": 8,  "demand_sig": 3.0},
        "Avionics":            {"n": int(n_skus * 0.15), "cost_range": (2_000, 80_000),
                                 "lead_range": (21,  90), "demand_mu": 2,  "demand_sig": 1.0},
        "Landing Gear":        {"n": int(n_skus * 0.10), "cost_range": (1_000, 40_000),
                                 "lead_range": (14,  45), "demand_mu": 4,  "demand_sig": 2.0},
        "Consumable":          {"n": int(n_skus * 0.20), "cost_range": (10,   500),
                                 "lead_range": (3,   14), "demand_mu": 50, "demand_sig": 15},
    }

    aircraft_types = ["B737-800", "A320neo", "B787-9", "A350-900", "B777-300ER"]
    records = []
    sku_counter = 1

    for cat, cfg in categories.items():
        n_cat = cfg["n"]
        for _ in range(n_cat):
            sku_id    = f"MRO-{sku_counter:04d}"
            unit_cost = float(rng.uniform(*cfg["cost_range"]))
            lead_time = int(rng.uniform(*cfg["lead_range"]))
            a_type    = rng.choice(aircraft_types)

            # Monthly demand: Poisson with slight seasonality
            base_demand = max(1, int(rng.normal(cfg["demand_mu"], cfg["demand_sig"])))
            season_mult = 1 + 0.15 * np.sin(np.linspace(0, 2 * np.pi, n_months))

            for month_idx in range(n_months):
                demand = max(0, int(rng.poisson(base_demand * season_mult[month_idx])))

                # Derived inventory metrics
                annual_demand    = demand * 12
                holding_cost_pct = 0.25  # 25% of unit cost per year (industry standard)
                reorder_cost     = 400.0 if cat == "Consumable" else 1500.0
                H                = unit_cost * holding_cost_pct
                D                = annual_demand + 1  # avoid /0
                eoq              = max(1, int(np.sqrt((2 * D * reorder_cost) / H)))
                sigma_demand     = cfg["demand_sig"]
                z_95             = 1.645
                safety_stock     = max(0, int(z_95 * sigma_demand * np.sqrt(lead_time / 30)))

                # ABC classification by annual spend
                annual_spend = annual_demand * unit_cost
                if annual_spend > 100_000:
                    abc = "A"
                elif annual_spend > 20_000:
                    abc = "B"
                else:
                    abc = "C"

                aog_critical = (cat in ["Engine Component", "Landing Gear"] and
                                unit_cost > 10_000)

                records.append({
                    "sku_id":           sku_id,
                    "part_name":        f"{cat} [{a_type}] #{sku_counter}",
                    "category":         cat,
                    "aircraft_type":    a_type,
                    "unit_cost_usd":    round(unit_cost, 2),
                    "lead_time_days":   lead_time,
                    "month_index":      month_idx + 1,
                    "monthly_demand":   demand,
                    "annual_demand":    annual_demand,
                    "holding_cost_pct": holding_cost_pct,
                    "reorder_cost_usd": reorder_cost,
                    "safety_stock":     safety_stock,
                    "eoq":             eoq,
                    "abc_class":        abc,
                    "aog_critical":     aog_critical,
                })
            sku_counter += 1

    df = pd.DataFrame(records)

    if save:
        out_path = SYNTH_DIR / "mro_synthetic_inventory.csv"
        df.to_csv(out_path, index=False)
        log.info(f"✅ Synthetic MRO data saved → {out_path} "
                 f"({df.shape[0]:,} rows, {n_skus} SKUs × {n_months} months)")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. UTILITY: LOAD ALL
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_datasets(skip_kaggle: bool = False) -> dict[str, pd.DataFrame]:
    """
    Convenience function to load all available datasets in one call.

    Parameters
    ----------
    skip_kaggle : bool
        If True, skips Kaggle downloads (useful when credentials aren't set up).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "aerospace_sc", "aviation_safety", "dataco", "bts", "mro_synthetic"
    """
    result = {}

    if not skip_kaggle:
        try:
            result["aerospace_sc"]    = load_aerospace_sc()
        except Exception as e:
            log.warning(f"Skipping aerospace_sc: {e}")

        try:
            result["aviation_safety"] = load_global_aviation_safety()
        except Exception as e:
            log.warning(f"Skipping aviation_safety: {e}")

        try:
            result["dataco"]          = load_dataco_supply_chain()
        except Exception as e:
            log.warning(f"Skipping dataco: {e}")

    result["bts"]           = load_bts_ontime()
    result["mro_synthetic"] = generate_mro_synthetic()

    log.info(f"📦 Loaded {len(result)} datasets: {list(result.keys())}")
    return result


# ── Quick test when run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== TESTING: Synthetic MRO Generation (no Kaggle needed) ===\n")
    df = generate_mro_synthetic(n_skus=50, n_months=12, save=False)
    print(f"Shape: {df.shape}")
    print(df.head(5).to_string())
    print(f"\nABC Distribution:\n{df.groupby('abc_class')['sku_id'].nunique()}")
    print(f"\nAOG Critical parts: {df[df['aog_critical']]['sku_id'].nunique()}")
    print("\n✅ ingest.py self-test passed.")
