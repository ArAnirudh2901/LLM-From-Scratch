"""Rescue rows from a gzip file on Colab's Drive mount that never finished uploading (the session died): read every
complete line (a damaged end is tolerated), save them as a new part in the parts folder and as /content/rescue.jsonl.gz.

    python -m kinlogic.rescue_drive [src] [parts_folder]      then, in the notebook: drive.flush_and_unmount()
"""
import gzip
import json
import os
import shutil
import sys
import zlib

src = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/kinthread/reworded_local.jsonl.gz'
parts = sys.argv[2] if len(sys.argv) > 2 else '/content/drive/MyDrive/kinthread/reworded_parts'
print('mounted:', os.path.ismount('/content/drive'), '| source exists:', os.path.exists(src),
      '| size:', os.path.getsize(src) if os.path.exists(src) else 0)
data, raw = b'', open(src, 'rb').read() if os.path.exists(src) else b''
while raw:   # one gzip member after another; stop quietly at a damaged one
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        data += d.decompress(raw)
    except zlib.error:
        break
    raw = d.unused_data
rows = []
for line in data.decode('utf-8', 'ignore').splitlines():
    try:
        json.loads(line)
        rows.append(line + '\n')
    except ValueError:
        pass
print(len(rows), 'complete rows recovered')
with gzip.open('/content/rescue.jsonl.gz', 'wt') as f:
    f.writelines(rows)
os.makedirs(parts, exist_ok=True)
shutil.copy('/content/rescue.jsonl.gz', os.path.join(parts, 'part-rescue.jsonl.gz'))
print('saved to', os.path.join(parts, 'part-rescue.jsonl.gz'))
