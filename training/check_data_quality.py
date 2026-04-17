"""Analyze the Sinhala TTS dataset for data quality issues."""
import os

METADATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metadata.csv")

speakers = {}
text_lengths = []
duplicates = {}
short_texts = []
long_texts = []
empty_texts = []
char_set = set()

with open(METADATA, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            print(f"  WARNING: Line {i} has only {len(parts)} fields: {line[:80]}")
            continue

        wav_id, roman_text, sinhala_text, speaker = parts[0], parts[1], parts[2], parts[3]

        speakers[speaker] = speakers.get(speaker, 0) + 1

        tlen = len(roman_text)
        text_lengths.append(tlen)
        char_set.update(roman_text)

        if tlen < 5:
            short_texts.append((wav_id, roman_text))
        if tlen > 300:
            long_texts.append((wav_id, tlen, roman_text[:60]))
        if not roman_text.strip():
            empty_texts.append(wav_id)

        if roman_text in duplicates:
            duplicates[roman_text].append(wav_id)
        else:
            duplicates[roman_text] = [wav_id]

# Results
print("=" * 60)
print("SINHALA TTS DATASET ANALYSIS")
print("=" * 60)

total = sum(speakers.values())
print(f"\nTotal samples: {total}")

print(f"\nSpeaker distribution:")
for s, c in sorted(speakers.items()):
    pct = c / total * 100
    print(f"  {s}: {c} samples ({pct:.1f}%)")

print(f"\nText length stats (romanized):")
text_lengths.sort()
print(f"  Min:    {min(text_lengths)} chars")
print(f"  Max:    {max(text_lengths)} chars")
print(f"  Mean:   {sum(text_lengths)/len(text_lengths):.1f} chars")
print(f"  Median: {text_lengths[len(text_lengths)//2]} chars")
print(f"  P10:    {text_lengths[int(len(text_lengths)*0.1)]} chars")
print(f"  P90:    {text_lengths[int(len(text_lengths)*0.9)]} chars")

# Estimated audio duration (rough: ~15 chars/sec for Sinhala speech)
est_total_sec = sum(text_lengths) / 12  # conservative estimate
print(f"\nEstimated total audio: ~{est_total_sec/3600:.1f} hours (rough, based on text length)")

print(f"\nUnique characters in romanized text: {len(char_set)}")
sorted_chars = sorted(char_set)
print(f"  {''.join(sorted_chars)}")

print(f"\n{'='*60}")
print("POTENTIAL ISSUES")
print(f"{'='*60}")

print(f"\n1. Empty texts: {len(empty_texts)}")
if empty_texts:
    for wid in empty_texts[:5]:
        print(f"   {wid}")

print(f"\n2. Very short texts (<5 chars): {len(short_texts)}")
if short_texts:
    for wid, txt in short_texts[:10]:
        print(f"   {wid}: \"{txt}\"")

print(f"\n3. Very long texts (>300 chars): {len(long_texts)}")
if long_texts:
    for wid, tl, txt in long_texts[:10]:
        print(f"   {wid}: {tl} chars")

dups = {k: v for k, v in duplicates.items() if len(v) > 1}
total_dup_samples = sum(len(v) for v in dups.values()) if dups else 0
print(f"\n4. Duplicate transcriptions: {len(dups)} texts appear multiple times ({total_dup_samples} total samples)")
if dups:
    for txt, ids in sorted(dups.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"   [{len(ids)}x] \"{txt[:70]}\"")
        print(f"         IDs: {', '.join(ids)}")

# Speaker balance
if len(speakers) > 1:
    counts = list(speakers.values())
    ratio = max(counts) / min(counts)
    print(f"\n5. Speaker balance ratio: {ratio:.1f}:1", end="")
    if ratio > 3:
        print(" ⚠️  IMBALANCED (>3:1)")
    else:
        print(" ✓ OK")

print(f"\n{'='*60}")
print("SUMMARY & RECOMMENDATIONS")
print(f"{'='*60}")

issues = []
if len(empty_texts) > 0:
    issues.append(f"Remove {len(empty_texts)} empty-text samples")
if len(short_texts) > 0:
    issues.append(f"Review {len(short_texts)} very short text samples (<5 chars)")
if len(long_texts) > 0:
    issues.append(f"Review {len(long_texts)} very long samples (>300 chars) - may need splitting")
if total_dup_samples > 0:
    issues.append(f"Remove or deduplicate {total_dup_samples} duplicate transcriptions")
if len(speakers) > 1:
    counts = list(speakers.values())
    ratio = max(counts) / min(counts)
    if ratio > 3:
        issues.append(f"Speaker data is imbalanced ({ratio:.1f}:1) - may hurt multi-speaker quality")

if issues:
    print("\nIssues found:")
    for j, issue in enumerate(issues, 1):
        print(f"  {j}. {issue}")
else:
    print("\nNo major metadata issues found!")

print(f"\nIMPORTANT: Run this on the SERVER to also check:")
print(f"  - Audio file durations (short <1s, long >15s)")
print(f"  - Missing audio files")
print(f"  - Audio quality (SNR, silence ratio)")
print(f"  - Volume consistency")
