import cv2
import numpy as np
from scipy.spatial import Delaunay
import os

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(param, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(param, f"{len(points)}", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow("Image", param)

def get_feature_points(img, window_name="Image"):
    global points
    points = []
    clone = img.copy()
    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, click_event, clone)
    print(f"Click on feature points in the {window_name} window. Press 'q' to finish.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        # press 'q' to quit collecting points
        if key == ord("q"):
            break
    cv2.destroyWindow(window_name)
    return points.copy()

def add_boundary_points(pts, w, h):
    pts.extend([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])
    return pts

def compute_delaunay(pts, w, h):
    # Using scipy.spatial.Delaunay on the points (as numpy array)
    pts_np = np.array(pts)
    tri = Delaunay(pts_np)
    # Each triangle is given by indices into pts_np
    return tri.simplices

def barycentric_coords(pt, tri_pts):
    x, y = pt
    (x1, y1), (x2, y2), (x3, y3) = tri_pts
    # Compute areas (using determinants)
    denom = ((y2 - y3)(x1 - x3) + (x3 - x2)(y1 - y3))
    if denom == 0:
        return (0, 0, 0)
    u = ((y2 - y3)(x - x3) + (x3 - x2)(y - y3)) / denom
    v = ((y3 - y1)(x - x3) + (x1 - x3)(y - y3)) / denom
    w = 1 - u - v
    return (u, v, w)

def is_inside_triangle(bary):
    u, v, w = bary
    # Allow a small epsilon tolerance.
    return (u >= -1e-4) and (v >= -1e-4) and (w >= -1e-4)

def morph_triangle(src, dst, out_img, src_tri, dst_tri, inter_tri, t):
    # Compute bounding box for the intermediate triangle
    inter_tri_np = np.array(inter_tri, dtype=np.int32)
    r = cv2.boundingRect(inter_tri_np)
    x, y, w, h = r

    for i in range(y, y + h):
        for j in range(x, x + w):
            pt = (j, i)
            bary = barycentric_coords(pt, inter_tri)
            if is_inside_triangle(bary):
                # Get corresponding points in source and destination triangles using barycentrics
                u, v, w_b = bary
                src_pt = np.array(src_tri[0]) * u + np.array(src_tri[1]) * v + np.array(src_tri[2]) * w_b
                dst_pt = np.array(dst_tri[0]) * u + np.array(dst_tri[1]) * v + np.array(dst_tri[2]) * w_b

                # Bilinear interpolation for colors:
                # For simplicity, we use nearest neighbor sampling here.
                src_color = src[int(round(src_pt[1])), int(round(src_pt[0]))]
                dst_color = dst[int(round(dst_pt[1])), int(round(dst_pt[0]))]
                color = (1 - t) * src_color + t * dst_color
                out_img[i, j] = color

def morph_images(src, dst, src_points, dst_points, tri_indices, n):
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    morphed_images = []
    h, w = src.shape[:2]

    # Append the original source image as the first frame.
    morphed_images.append(src.astype(np.uint8))

    # Generate intermediate images for i=1 to n.
    for i in range(1, n + 1):
        t = i / (n + 1)
        # Compute intermediate feature points: P(t) = (1-t)*P_src + t*P_dst
        inter_points = []
        for p_src, p_dst in zip(src_points, dst_points):
            x = (1 - t) * p_src[0] + t * p_dst[0]
            y = (1 - t) * p_src[1] + t * p_dst[1]
            inter_points.append((x, y))

        # Create an empty output image.
        inter_img = np.zeros((h, w, 3), dtype=np.float32)

        # For each triangle, warp and blend
        for tri in tri_indices:
            idx1, idx2, idx3 = tri
            src_tri = [src_points[idx1], src_points[idx2], src_points[idx3]]
            dst_tri = [dst_points[idx1], dst_points[idx2], dst_points[idx3]]
            inter_tri = [inter_points[idx1], inter_points[idx2], inter_points[idx3]]
            morph_triangle(src, dst, inter_img, src_tri, dst_tri, inter_tri, t)
        # Clip values and convert to 8-bit.
        inter_img = np.clip(inter_img, 0, 255).astype(np.uint8)
        morphed_images.append(inter_img)

    # Append the original destination image as the last frame.
    morphed_images.append(dst.astype(np.uint8))
    return morphed_images

# Read the two input images (source and destination)
src_path = r"Mani.jpg"
dst_path = r"Anuj.jpg"
src_img = cv2.imread(src_path)
dst_img = cv2.imread(dst_path)
if src_img is None or dst_img is None:
    raise FileNotFoundError("One or both input images were not found. Check the paths.")
h, w = src_img.shape[:2]
dst_img = cv2.resize(dst_img, (w, h))
src_points = get_feature_points(src_img.copy(), "Source Image")
dst_points = get_feature_points(dst_img.copy(), "Destination Image")
src_points = add_boundary_points(src_points, w, h)
dst_points = add_boundary_points(dst_points, w, h)
if len(src_points) != len(dst_points):
    raise ValueError("The number of feature points in source and destination images must be the same.")
tri_indices = compute_delaunay(src_points, w, h)
n = 5 
morphed_images = morph_images(src_img, dst_img, src_points, dst_points, tri_indices, n)
output_folder = "morphed_sequence"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for i, img in enumerate(morphed_images):
    output_path = os.path.join(output_folder, f"morph_{i}.jpg")
    cv2.imwrite(output_path, img)
    print(f"Saved {output_path}")
print("Morphing sequence generated successfully.")
for img in morphed_images:
    cv2.imshow("Morphed Image", img)
    cv2.waitKey(500)
cv2.destroyAllWindows()