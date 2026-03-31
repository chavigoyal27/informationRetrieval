#Annotation Sheet Extractor
#randomly samples 1000 records from corpus_balanced_1to1.csv
#exports CSV

import csv
import random
import os

random.seed(42)  # fixed seed so everyone gets the same 1000 records

INPUT_FILE = "corpus_balanced_1to1.csv"   
OUTPUT_FILE = "annotation_sheet.csv"

SAMPLE_SIZE = 1000

def main():
    print("Reading corpus...")

    with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total records available: {len(rows):,}")

    # Sample 1000 records
    sampled = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

    print(f"Sampled {len(sampled)} records")

    # Show source distribution of sample
    source_counts = {}
    for r in sampled:
        src = r.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print("\nSource distribution in sample:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<12} {count}")

    # Write annotation sheet
    fieldnames = ["id", "source", "text", "annotator1_label", "annotator2_label", "annotator3_label", "final_label"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sampled:
            writer.writerow({
                "id": r["id"],
                "source": r["source"],
                "text": r["text"],
                "annotator1_label": "",       # fill in your own column only
                "annotator2_label": "",
                "annotator3_label": "",
                "final_label": "",       # filled in after comparing — majority vote
            })

    print(f"\nDone! Saved to: {OUTPUT_FILE}")
    print(f"Upload {OUTPUT_FILE} to Google Sheets and share with your team.")

if __name__ == "__main__":
    main()