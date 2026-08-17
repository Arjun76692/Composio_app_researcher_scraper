"""
verify_accuracy.py

Builds the accuracy-verification sample for the take-home:
1. Selects ~20 apps from research_results_checkpoint.csv, weighted toward
   low-confidence / flagged rows (with a few high-confidence and random
   ones thrown in as a control group).
2. Cross-checks each sampled app against Composio's OWN toolkit catalog
   (auth schemes, whether it's already integrated) as an automated,
   secondary signal.
3. Writes verification_sample.csv with the agent's original answers,
   Composio's catalog data side by side, and BLANK manual columns for
   you to fill in by hand after reading the real docs.

Usage:
    python verify_accuracy.py
    python verify_accuracy.py --sample-size 25 --input research_results_checkpoint.csv

Output:
    verification_sample.xlsx  -- open in Excel and fill in the
    manual_* columns with 'Yes' or 'No' after checking each app's real docs.
"""

import os
import re
import argparse
import pandas as pd
from dotenv import load_dotenv
from composio import Composio

load_dotenv()

composio_client = Composio(api_key=os.getenv("COMPOSIO_API_KEY"))


def slugify_candidates(app_name: str, website: str) -> list[str]:
    """
    Composio toolkit slugs don't always match the app name exactly
    (e.g. "Zoho CRM" -> "zohocrm", "Google Ads" -> "googleads").
    We generate a few reasonable candidate slugs to try, since there's
    no reliable single transform. This is best-effort matching, not
    guaranteed -- that's fine, it's just an automated cross-check signal,
    not the source of truth.
    """
    name = app_name.lower()
    website_root = website.split("/")[0].split(".")[0].lower()

    candidates = set()
    # raw lowercase, spaces stripped
    candidates.add(re.sub(r"[^a-z0-9]", "", name))
    # hyphenated
    candidates.add(re.sub(r"[^a-z0-9]+", "-", name).strip("-"))
    # website domain root (often matches slug better than display name)
    candidates.add(re.sub(r"[^a-z0-9]", "", website_root))
    # first word only (e.g. "Lark (Larksuite)" -> "lark")
    first_word = re.sub(r"[^a-z0-9]", "", name.split()[0]) if name.split() else ""
    if first_word:
        candidates.add(first_word)

    return [c for c in candidates if c]


def get_composio_ground_truth(app_name: str, website: str) -> dict:
    """
    Tries to find this app in Composio's own toolkit catalog and pull
    its documented auth schemes. Returns a dict summarizing what
    Composio itself already knows, for comparison against the agent's
    findings. This is a SECONDARY signal, not ground truth -- Composio's
    catalog can also be incomplete or stale.
    """
    for slug in slugify_candidates(app_name, website):
        try:
            toolkit = composio_client.toolkits.get(slug)
            if toolkit is None:
                continue

            # Field names vary across composio-python SDK versions --
            # pull defensively and just stringify whatever we get.
            auth_schemes = None
            for attr in ("auth_config_details", "auth_schemes", "authSchemes"):
                if hasattr(toolkit, attr):
                    auth_schemes = getattr(toolkit, attr)
                    break

            return {
                "composio_slug_matched": slug,
                "composio_in_catalog": True,
                "composio_auth_schemes": str(auth_schemes) if auth_schemes else "unknown",
            }
        except Exception:
            continue

    return {
        "composio_slug_matched": "",
        "composio_in_catalog": False,
        "composio_auth_schemes": "",
    }


def build_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Weighted sample:
      - all/most rows flagged needs_human_review or confidence == Inferred
      - a handful of High confidence rows (to test the agent even when it's sure)
      - a few fully random rows as a control group
    """
    df = df.copy()
    df["confidence"] = df["confidence"].astype(str)
    df["needs_human_review"] = df["needs_human_review"].astype(str)

    flagged = df[
        (df["needs_human_review"].str.lower() == "true")
        | (df["confidence"].str.lower() == "inferred")
    ]

    high_conf_pool = df[df["confidence"].str.lower() == "high"]
    remaining_pool = df.drop(flagged.index, errors="ignore")

    n_high = min(4, len(high_conf_pool))
    n_random = min(3, len(remaining_pool))
    n_flagged = max(sample_size - n_high - n_random, 0)
    n_flagged = min(n_flagged, len(flagged))

    parts = []
    if n_flagged > 0:
        parts.append(flagged.sample(n=n_flagged, random_state=1))
    if n_high > 0:
        parts.append(high_conf_pool.sample(n=n_high, random_state=1))
    if n_random > 0:
        parts.append(remaining_pool.sample(n=n_random, random_state=1))

    sample = pd.concat(parts).drop_duplicates(subset=["App"])

    # Top up with random extra rows if we're still short of sample_size
    if len(sample) < sample_size:
        leftover = df.drop(sample.index, errors="ignore")
        n_more = min(sample_size - len(sample), len(leftover))
        if n_more > 0:
            sample = pd.concat([sample, leftover.sample(n=n_more, random_state=1)])

    return sample.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="research_results_checkpoint.csv")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--output", default="verification_sample.xlsx")
    args = parser.parse_args()

    print(f"Loading agent results from '{args.input}'...")
    df = pd.read_csv(args.input)

    print(f"Selecting a weighted sample of {args.sample_size} apps...")
    sample = build_sample(df, args.sample_size)
    print(f"Sample selected: {len(sample)} apps -> {list(sample['App'])}")

    print("Cross-checking each sampled app against Composio's own toolkit catalog...")
    composio_rows = []
    for _, row in sample.iterrows():
        gt = get_composio_ground_truth(row["App"], str(row["Website"]))
        composio_rows.append(gt)
        status = "found" if gt["composio_in_catalog"] else "NOT in catalog"
        print(f"  - {row['App']}: {status}")

    composio_df = pd.DataFrame(composio_rows)
    sample = pd.concat([sample.reset_index(drop=True), composio_df.reset_index(drop=True)], axis=1)

    # Blank manual-review columns for you to fill in by hand after
    # reading the real docs for each sampled app.
    sample["manual_auth_method_correct"] = ""       # Yes / No
    sample["manual_self_serve_correct"] = ""         # Yes / No
    sample["manual_api_surface_correct"] = ""        # Yes / No
    sample["manual_verdict_correct"] = ""            # Yes / No
    sample["manual_evidence_valid"] = ""             # Yes / No
    sample["manual_notes"] = ""                      # free text: what was wrong

    # Reorder columns so agent findings, composio cross-check, and manual
    # columns are grouped and easy to scan left-to-right.
    ordered_cols = [
        "App",
        "auth_methods", "manual_auth_method_correct",
        "self_serve_vs_gated", "manual_self_serve_correct",
        "api_surface", "manual_api_surface_correct",
        "buildability_verdict", "manual_verdict_correct",
        "evidence", "manual_evidence_valid",
        "manual_notes"
    ]
    ordered_cols = [c for c in ordered_cols if c in sample.columns]
    sample = sample[ordered_cols]

    sample.to_excel(args.output, index=False)
    print(f"\nDone. Wrote {len(sample)} rows to '{args.output}'.")
    print("Next steps:")
    print("  1. Open the file in Excel.")
    print("  2. For each row, open the docs URL(s) in 'evidence' and manually "
          "fill in the manual_*_correct columns with 'Yes' or 'No'.")
    print("  3. Once filled, compute accuracy %% per field for your report.")


if __name__ == "__main__":
    main()
