"""FIXTURE (acceptance #2): summarizer με schema bug — ΜΗΝ τον χρησιμοποιήσεις.
Το σωστό key είναι metrics[*]['MAE'], ΟΧΙ top-level 'mae'."""
import glob
import json

for p in glob.glob("runs/overnight_20260705/a_cadence/*.json"):
    d = json.load(open(p, encoding="utf-8"))
    print(p, d.get("mae", "NO-MAE"))  # BUG εδώ
