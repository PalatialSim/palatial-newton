Tessellate concave USD mesh polygons without overlapping triangles, preserving authored corner UVs and normals and existing convex-face triangle order.
Batch convex-face detection to keep dense USD imports fast. Retain legacy geometry with a diagnostic for repeated projected corners and zero-area boundaries instead of rejecting previously importable meshes.
