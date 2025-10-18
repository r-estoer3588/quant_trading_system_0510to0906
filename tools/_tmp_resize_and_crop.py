from PIL import Image

p_in = r"docs/eyecatch_candidates/week1_candidate.png"
p_out = r"docs/eyecatch_candidates/week1_candidate_resized.png"
img = Image.open(p_in)
print("orig size:", img.size, "mode:", img.mode)
w, h = img.size
target_w, target_h = 1280, 670
# Calculate scale to cover target
scale = max(target_w / w, target_h / h)
new_w = int(w * scale)
new_h = int(h * scale)
img = img.resize((new_w, new_h), Image.LANCZOS)
left = (new_w - target_w) // 2
top = (new_h - target_h) // 2
img = img.crop((left, top, left + target_w, top + target_h))
img.save(p_out)
print("saved:", p_out)
