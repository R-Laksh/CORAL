"""Verify a BPNet window against ENCODE's released chromosome split archive."""
import hashlib
import json
from pathlib import Path
import tarfile


REGION_ARCHIVE_SHA256 = "2a3d65f4939e2b77a161342be54d3f011bf0d6010eb7bc933b56d13e503c0db7"


def verify_fold_window(archive, window, fold=0, experiment="ENCSR865RXA",
                       expected_sha256=REGION_ARCHIVE_SHA256):
    """Read exact tar members without extracting paths or executable content."""
    path = Path(archive)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("ENCODE split archive checksum mismatch")
    chrom, start, end = window["chrom"], int(window["start"]), int(window["end"])
    if window["genome"] != "hg38" or end - start != 2114:
        raise ValueError("Expected a 2114bp hg38 window")
    with tarfile.open(path) as handle:
        read = lambda name: handle.extractfile(f"./fold_{fold}/{name}").read().decode()
        split = json.loads(read(f"cv_params.fold_{fold}.json"))
        if chrom not in split["test"] or chrom in split["train"] + split["valid"]:
            raise ValueError("Window chromosome is not exclusively held out for this fold")
        overlap, matching_peak, members = {}, [], {}
        for group in ("training", "validation", "test"):
            for kind in ("peaks", "nonpeaks"):
                member = f"{kind}.{group}set.fold_{fold}.{experiment}.bed"
                rows = [line.split("\t") for line in read(member).splitlines() if line]
                hits = [r for r in rows if r[0] == chrom and int(r[1]) < end and int(r[2]) > start]
                key = f"{kind}_{group}"
                overlap[key] = len(hits)
                members[key] = {"member": f"./fold_{fold}/{member}", "rows": len(rows)}
                if kind == "peaks" and group == "test":
                    matching_peak = [r for r in hits if int(r[1]) + int(r[9]) == start + 1057]
        if any(overlap[f"{kind}_{group}"] for kind in ("peaks", "nonpeaks")
               for group in ("training", "validation")):
            raise ValueError("Window overlaps a training or validation region")
        if len(matching_peak) != 1:
            raise ValueError("Window summit does not uniquely match a released test peak")
    return {"status": "verified_test_window", "fold": fold, "experiment": experiment,
            "archive_accession": "ENCFF277YRG", "archive_sha256": digest,
            "archive_url": "https://www.encodeproject.org/files/ENCFF277YRG/",
            "chromosome_split": split, "window": {k: window[k] for k in ("genome", "chrom", "start", "end")},
            "overlap_counts": overlap, "members": members, "matching_test_peak": matching_peak[0],
            "scope": "Reference window is out of fold; mutant occupancy remains unmeasured"}
