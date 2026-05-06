"""Split a multi-object VHACD OBJ into individual OBJ files."""
import os, sys

src = os.path.join(os.path.dirname(__file__), "torus_vhacd.obj")
out_dir = os.path.dirname(__file__)

current_name = None
verts = []
faces = []
vert_offset = 0
files_written = []

def flush(name, verts, faces, out_dir):
    if not name or not verts:
        return
    path = os.path.join(out_dir, f"torus_vhacd_{name}.obj")
    with open(path, "w") as f:
        for v in verts:
            f.write(v + "\n")
        for face in faces:
            f.write(face + "\n")
    return path

with open(src) as f:
    for line in f:
        line = line.rstrip()
        if line.startswith("o "):
            if current_name:
                p = flush(current_name, verts, faces, out_dir)
                if p: files_written.append(p)
                vert_offset += len(verts)
            current_name = line[2:].strip()
            verts = []
            faces = []
        elif line.startswith("v "):
            verts.append(line)
        elif line.startswith("f "):
            parts = line.split()
            new_parts = ["f"]
            for p in parts[1:]:
                idx = int(p.split("/")[0]) - vert_offset
                new_parts.append(str(idx))
            faces.append(" ".join(new_parts))

if current_name:
    p = flush(current_name, verts, faces, out_dir)
    if p: files_written.append(p)

for fp in files_written:
    print(f"  wrote: {os.path.basename(fp)}")
print(f"Total: {len(files_written)} files")
