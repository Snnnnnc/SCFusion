import numpy as np
import pandas as pd
from collections import Counter
import os
import re

bdf_path = "/Users/shennc/Desktop/THU/25 SPR/comfort sim/motion_sickness_classification/data/20251109195509_whj_04_whj/20251109195509_whj_04_whj/whj/evt.bdf"

def read_annotations_bdf_raw(bdf_path):
    import mmap
    with open(bdf_path, 'rb') as f:
        # Check size first
        size = os.path.getsize(bdf_path)
        print(f"File size: {size} bytes")
        
        # Look for numbers in the file using a simple approach
        data = f.read()
        
        # BDF evt files store events in TAL (Time-stamped Annotation Lists)
        # We look for the marker 0x14
        events = []
        i = 0
        while i < len(data):
            if data[i] == 0x14:
                # Found an annotation marker
                # Read until next 0x14 or 0x00
                j = i + 1
                while j < len(data) and data[j] != 0x14 and data[j] != 0x00:
                    j += 1
                desc = data[i+1:j].decode('latin-1', errors='ignore')
                if desc and desc.isdigit():
                    events.append([0, 0, desc])
                i = j
            else:
                i += 1
        return events
        
    events = []
    for ev in triggers:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for description in ev[3].split('\x14')[1:]:
            if description:
                events.append([onset, duration, description])
    return events

try:
    print(f"Reading {bdf_path}...")
    events = read_annotations_bdf_raw(bdf_path)
    
    if not events:
        print("No events found using regex pattern.")
    else:
        print(f"Found {len(events)} events.")
        descriptions = [ev[2] for ev in events]
        counts = Counter(descriptions)
        print("\nEvent Description Distribution:")
        for desc, count in sorted(counts.items()):
            print(f"  Rating/Event {desc}: {count} times")
            
except Exception as e:
    print(f"Error reading BDF file: {e}")
