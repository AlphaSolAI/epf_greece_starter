import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings

# UTF-8 output για να μην βλέπεις ιερογλυφικά
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
HENEX_DIRS = sorted(RAW_DIR.glob("*_EL-DAM_Results"))  # 2020_EL-DAM_Results κλπ
LOAD_DIR = RAW_DIR / "load"

OUT_PATH = ROOT / "data" / "processed" / "hourly.parquet"


# ----------------- helpers ----------------- #
def _find_col(columns_map, keyword):
    """Βρες την πρώτη στήλη που περιέχει το keyword (case-insensitive)."""
    keyword = keyword.upper()
    for col, name in columns_map.items():
        if keyword in name:
            return col
    return None


def parse_henex_file(path: Path):
    """
    Διαβάζει ένα HenEx EL-DAM_Results xlsx και επιστρέφει:
        timestamp, price
    όπου timestamp είναι η DELIVERY_MTU και price το MCP.
    Γίνεται όσο πιο robust γίνεται.
    """
    try:
        df = pd.read_excel(path, engine="openpyxl")
        if df.empty:
            return None

        # Κανονικοποίηση ονομάτων στηλών
        col_map = {c: str(c).strip().upper() for c in df.columns}

        mcp_col = _find_col(col_map, "MCP")
        mtu_col = _find_col(col_map, "DELIVERY_MTU")
        zone_col = _find_col(col_map, "BIDDING_ZONE")
        dday_col = _find_col(col_map, "DDAY")

        if mcp_col is None or mtu_col is None:
            # Χωρίς MCP / DELIVERY_MTU δεν κάνουμε τίποτα
            return None

        # Φιλτράρισμα για Mainland Greece αν υπάρχει ζώνη
        if zone_col is not None:
            mask_zone = df[zone_col].astype(str).str.contains(
                "Mainland", case=False, na=False
            )
            df = df[mask_zone]

        # Τιμές
        prices = df[mcp_col]

        # MCP μπορεί να είναι με κόμμα ως δεκαδικό
        if prices.dtype == object:
            prices = (
                prices.astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )

        prices = pd.to_numeric(prices, errors="coerce")

        # Χρόνος
        ts = pd.to_datetime(df[mtu_col], errors="coerce")

        # Αν (για κάποιο λόγο) το DELIVERY_MTU είναι μόνο ώρα,
        # τότε συνθέτουμε από DDAY + ώρα
        if ts.isna().all() and dday_col is not None:
            days = pd.to_datetime(df[dday_col].astype(str), format="%Y%m%d", errors="coerce")
            times = pd.to_datetime(df[mtu_col].astype(str), errors="coerce").dt.time
            ts = pd.to_datetime(
                days.dt.strftime("%Y-%m-%d") + " " + pd.Series(times).astype(str),
                errors="coerce",
            )

        out = pd.DataFrame({"timestamp": ts, "price": prices})
        out = out.dropna(subset=["timestamp", "price"])

        if out.empty:
            return None

        # Κάποιες ώρες είναι 15λεπτες (00:00, 00:15 κλπ).
        # Παίρνουμε μέσο όρο ανά ώρα.
        out["timestamp_hour"] = out["timestamp"].dt.floor("H")
        out = (
            out.groupby("timestamp_hour")["price"]
            .mean()
            .reset_index()
            .rename(columns={"timestamp_hour": "timestamp"})
        )

        return out

    except Exception:
        return None


def parse_admie_load_hourly(path: Path):
    """
    Διαβάζει ένα αρχείο RealTimeSCADASystemLoad_01.xls (.xls) και φτιάχνει:
        timestamp, load
    από στήλες:
        Date | 01 | 02 | ... | 24
    Το mapping είναι: ώρα 1 -> 00:00, 2 -> 01:00, ..., 24 -> 23:00.
    """
    try:
        # Scan για να βρούμε τη γραμμή με το header
        df_scan = pd.read_excel(path, header=None, nrows=25, engine="xlrd")
        header_idx = -1
        for i, row in df_scan.iterrows():
            row_vals = [str(v).strip().lower() for v in row.values]
            if any("date" in x or "ημερ" in x for x in row_vals) and any(
                x == "1" or x == "1.0" for x in row_vals
            ):
                header_idx = i
                break

        if header_idx == -1:
            return None

        df = pd.read_excel(path, header=header_idx, engine="xlrd")
        df.columns = [str(c).strip() for c in df.columns]

        # στήλη ημερομηνίας
        date_col = None
        for c in df.columns:
            if "date" in c.lower() or "ημερ" in c.lower():
                date_col = c
                break
        if date_col is None:
            return None

        # στήλες με ώρες 1..24
        hour_cols = []
        for c in df.columns:
            try:
                val = float(c)
                if 1 <= val <= 24:
                    hour_cols.append(c)
            except Exception:
                continue

        if not hour_cols:
            return None

        # Καθαρισμός
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = df.dropna(subset=[date_col])

        for c in hour_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Wide -> Long
        long_df = df.melt(
            id_vars=[date_col],
            value_vars=hour_cols,
            var_name="hour",
            value_name="load",
        )

        long_df = long_df.dropna(subset=["load"])

        # hour: "1" -> 1, "1.0" -> 1
        long_df["hour"] = (
            long_df["hour"].astype(str).str.extract(r"(\d+)").astype(int)
        )

        # timestamp = date + (hour-1) hours
        long_df["timestamp"] = long_df[date_col] + pd.to_timedelta(
            long_df["hour"] - 1, unit="H"
        )

        out = long_df[["timestamp", "load"]].copy()
        out = out.dropna(subset=["timestamp", "load"])

        if out.empty:
            return None

        # merge πιθανών διπλοεγγραφών
        out = (
            out.groupby("timestamp")["load"]
            .mean()
            .reset_index()
            .sort_values("timestamp")
        )

        return out

    except Exception:
        return None


# ----------------- main ----------------- #
def main():
    print("⚙️ Building HOURLY dataset from HenEx + ADMIE...")

    # --- HenEx prices (hourly) --- #
    henex_files = []
    for d in HENEX_DIRS:
        henex_files.extend(sorted(d.glob("*.xls*")))

    print(f"   HenEx dirs: {[d.name for d in HENEX_DIRS]}")
    print(f"   Found {len(henex_files)} HenEx Excel files.")

    henex_parts = []
    for i, f in enumerate(henex_files):
        part = parse_henex_file(f)
        if part is not None:
            henex_parts.append(part)
        if (i + 1) % 200 == 0:
            print(f"      ...parsed {i+1} HenEx files (ok: {len(henex_parts)})")

    if not henex_parts:
        print("❌ No HenEx hourly data parsed. Check file pattern/structure.")
        return

    henex_all = pd.concat(henex_parts, ignore_index=True)
    henex_all = (
        henex_all.groupby("timestamp")["price"]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )

    print(
        f"   ✅ HenEx hourly prices: {len(henex_all)} rows "
        f"({henex_all['timestamp'].min()} → {henex_all['timestamp'].max()})"
    )

    # --- ADMIE load (hourly) --- #
    load_files = sorted(LOAD_DIR.glob("*.xls*"))
    print(f"   ADMIE load dir: {LOAD_DIR.name} | files: {len(load_files)}")

    load_parts = []
    for i, f in enumerate(load_files):
        part = parse_admie_load_hourly(f)
        if part is not None:
            load_parts.append(part)
        if (i + 1) % 200 == 0:
            print(f"      ...parsed {i+1} load files (ok: {len(load_parts)})")

    if not load_parts:
        print("❌ No ADMIE hourly load parsed. Check load files.")
        return

    load_all = pd.concat(load_parts, ignore_index=True)
    load_all = (
        load_all.groupby("timestamp")["load"]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )

    print(
        f"   ✅ ADMIE hourly load: {len(load_all)} rows "
        f"({load_all['timestamp'].min()} → {load_all['timestamp'].max()})"
    )

    # --- Merge price + load --- #
    merged = pd.merge(
        henex_all, load_all, on="timestamp", how="inner"
    ).sort_values("timestamp")

    print(f"   🔗 After INNER JOIN: {len(merged)} hourly rows.")

    # keep μόνο τις στήλες που θέλουμε προς το παρόν
    merged = merged.set_index("timestamp")
    merged.index.name = "timestamp"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH)

    print("✅ Saved hourly dataset:")
    print(f"   {OUT_PATH}")
    print(merged.head(10))
    print(merged.tail(10))


if __name__ == "__main__":
    main()
