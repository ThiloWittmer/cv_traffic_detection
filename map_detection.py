import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import img_as_bool
from igraph import Graph
from skimage.graph import route_through_array
from sklearn.cluster import DBSCAN
import json
from enums import Direction


def find_endpoints_and_junctions(skel_img):
    skel = skel_img.copy()
    if skel.dtype != np.uint8:
        skel = (skel * 255).astype(np.uint8)

    endpoints = []
    junctions = []

    for y in range(1, skel.shape[0] - 1):
        for x in range(1, skel.shape[1] - 1):
            if skel[y, x] == 0:
                continue

            roi = skel[y - 1:y + 2, x - 1:x + 2]
            count = cv.countNonZero(roi) - 1  # ohne Mittelpunkt

            if count == 1:
                endpoints.append((x, y))
            elif count >= 3:
                junctions.append((x, y))

    return endpoints, junctions


def cluster_points(points, eps=7):
    if not points:
        return []

    X = np.array(points)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(X)

    labels = clustering.labels_
    unique_labels = set(labels)

    clustered_points = []
    for label in unique_labels:
        cluster_members = X[labels == label]
        centroid = np.mean(cluster_members, axis=0)
        clustered_points.append(tuple(centroid.astype(int)))

    return clustered_points

def nearest_skel_point(coord, skel):
    y, x = coord[1], coord[0]
    if skel[y, x]:
        return (x, y)
    # Suche im Umkreis nach dem nächsten Skeleton-Pixel
    coords = np.column_stack(np.where(skel))
    dists = np.sqrt((coords[:, 0] - y)**2 + (coords[:, 1] - x)**2)
    idx = np.argmin(dists)
    return (coords[idx][1], coords[idx][0])

def find_colored_center(img, lower, upper):
    """Findet das Zentrum der größten Fläche im Farbbereich (HSV)."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv.contourArea)
    M = cv.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def insert_special_node(g, layout, skel, special_coord):
    """Fügt einen Knoten für Start/Ziel ein und splittet ggf. eine Kante."""
    from skimage.graph import route_through_array

    # 1. Projiziere auf Skeleton
    special_on_skel = nearest_skel_point(special_coord, skel)
    # 2. Finde nächste Kante im Graphen (kürzester Pfad, der den Punkt enthält)
    min_weight = np.inf
    best_edge = None
    best_path = None
    for edge in g.es:
        src, tgt = edge.tuple
        src_pt = nearest_skel_point(layout[src], skel)
        tgt_pt = nearest_skel_point(layout[tgt], skel)
        cost = (~skel).astype(np.float32)
        cost[cost == 1] = 1000
        try:
            indices, weight = route_through_array(
                cost,
                start=(src_pt[1], src_pt[0]),
                end=(tgt_pt[1], tgt_pt[0]),
                fully_connected=True
            )
            path_coords = np.array([(p[1], p[0]) for p in indices])
            # Prüfe, ob der Punkt auf dem Pfad liegt (Toleranz 2 Pixel)
            dists = np.sqrt((path_coords[:,0] - special_on_skel[0])**2 + (path_coords[:,1] - special_on_skel[1])**2)
            if np.any(dists <= 2) and weight < min_weight:
                min_weight = weight
                best_edge = edge
                best_path = indices
        except Exception:
            continue
    if best_edge is None:
        print("Kein passender Pfad für Start/Ziel gefunden!")
        return None, layout
    # 3. Splitte die Kante: entferne alte, füge neuen Knoten und zwei neue Kanten ein
    src, tgt = best_edge.tuple
    g.delete_edges([best_edge.index])
    new_idx = g.vcount()
    g.add_vertex()
    layout.append(special_coord)
    # Pfad von src bis special
    cost = (~skel).astype(np.float32)
    cost[cost == 1] = 1000
    indices1, weight1 = route_through_array(
        cost,
        start=(nearest_skel_point(layout[src], skel)[1], nearest_skel_point(layout[src], skel)[0]),
        end=(special_on_skel[1], special_on_skel[0]),
        fully_connected=True
    )
    indices2, weight2 = route_through_array(
        cost,
        start=(special_on_skel[1], special_on_skel[0]),
        end=(nearest_skel_point(layout[tgt], skel)[1], nearest_skel_point(layout[tgt], skel)[0]),
        fully_connected=True
    )
    g.add_edge(src, new_idx, weight=weight1)
    g.add_edge(new_idx, tgt, weight=weight2)
    return new_idx, layout

def get_turn_sequence(layout, path) -> list[Direction]:
    directions: list[Direction] = []
    for i in range(1, len(path)-1):
        prev = np.array(layout[path[i-1]])
        curr = np.array(layout[path[i]])
        nex = np.array(layout[path[i+1]])
        v1 = curr - prev
        v2 = nex - curr
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angle = (angle + np.pi) % (2 * np.pi) - np.pi  # auf [-pi, pi]
        if abs(angle) < np.pi/6:
            directions.append(Direction.GERADEAUS)
        elif angle < 0:
            directions.append(Direction.LINKS)
        else:
            directions.append(Direction.RECHTS)
    return directions

def detect_turns(img_path: str) -> list[Direction] | None:
    img = cv.imread(img_path)
    if img is not None:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    else:
        print("konnte bild in detect_turns nicht laden")
        return None

    canny1 = cv.Canny(gray, 50, 200, None, 3)

    cdst = cv.cvtColor(canny1, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(canny1, 1, np.pi / 720, 50, None, 100, 42)


    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    kernel_large = np.ones((25,25),np.uint8)
    kernel_small = np.ones((5,5),np.uint8)

    cdst = cv.morphologyEx(cdst, cv.MORPH_CLOSE, kernel_large)
    cdst = cv.erode(cdst, kernel_small)

    binary = cv.threshold(cv.cvtColor(cdst, cv.COLOR_BGR2GRAY),30, 255, cv.THRESH_BINARY)[1]
    bin_bool = img_as_bool(binary)

    skel = skeletonize(bin_bool)
    skel = skel.astype(bool)

    endpoints, junctions = find_endpoints_and_junctions(skel)
    nodes = cluster_points(endpoints + junctions,eps=30)

    coord_to_index = {coord: idx for idx, coord in enumerate(nodes)}
    index_to_coord = {idx: coord for coord, idx in coord_to_index.items()}

    g = Graph()
    g.add_vertices(len(nodes))


    layout = [index_to_coord[i] for i in range(len(nodes))]
    # Vor dem Pfadfinden:
    skel_node_pixels = [nearest_skel_point(coord, skel) for coord in layout]

    for i, coord_start in index_to_coord.items():
        for j, coord_end in index_to_coord.items():
            if i >= j:
                continue
            start = nearest_skel_point(coord_start, skel)
            end = nearest_skel_point(coord_end, skel)
            try:
                cost = (~skel).astype(np.float32)
                cost[cost == 1] = 1000  # unbegehbar

                indices, weight = route_through_array(
                    cost,
                    start=(start[1], start[0]),
                    end=(end[1], end[0]),
                    fully_connected=True
                )

                # Prüfe, ob auf dem Pfad weitere Knoten (außer Start/Ziel) liegen
                path_coords = np.array([(p[1], p[0]) for p in indices])
                is_direct = True
                for k, other in enumerate(skel_node_pixels):
                    if k == i or k == j:
                        continue
                    # Euklidische Distanz zu allen Pfadpunkten prüfen
                    dists = np.sqrt((path_coords[:,0] - other[0])**2 + (path_coords[:,1] - other[1])**2)
                    if np.any(dists <= 15):
                        is_direct = False
                        break

                if is_direct and weight < 1e2:
                    g.add_edge(i, j, weight=weight)

            except Exception as e:
                print(f"Fehler Pfad: {start} : {end} : {e}")

    # In (x, y) umwandeln für matplotlib
    x = [p[0] for p in layout]
    y = [p[1] for p in layout]

    plt.subplot(1,3,1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap='gray')
    plt.title('original')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(cdst, cmap='gray')
    plt.title('hl')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,3)
    plt.imshow(binary, cmap='gray')
    plt.title('bin')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.imshow(skel, cmap='gray')
    plt.title('skel')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Zeichne
    plt.figure(figsize=(8, 8))

    # Kanten zeichnen
    for edge in g.es:
        src = layout[edge.source]
        tgt = layout[edge.target]
        plt.plot([src[0], tgt[0]], [src[1], tgt[1]], 'b-', linewidth=1)
    plt.scatter(x, y, c='red', s=50, label='Knoten')
    plt.title("Straßen-Graph aus Skeleton")
    plt.gca().invert_yaxis()  
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap='gray')  # Bild im Hintergrund

    # Kanten zeichnen
    for edge in g.es:
        src, tgt = edge.tuple
        x0, y0 = layout[src]
        x1, y1 = layout[tgt]
        plt.plot([x0, x1], [y0, y1], color='y', linewidth=3)

    # Knoten zeichnen
    for idx, (x, y) in enumerate(layout):
        plt.scatter(x, y, color='red', s=50)


    plt.title("Graph über Stadtplan")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Farbgrenzen für Start (grün) und Ziel (rot) im HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    start_coord = find_colored_center(img, lower_green, upper_green)
    ziel_coord = find_colored_center(img, lower_red1, upper_red1)
    if ziel_coord is None:
        ziel_coord = find_colored_center(img, lower_red2, upper_red2)

    if start_coord and ziel_coord:
        start_idx, layout = insert_special_node(g, layout, skel, start_coord)
        ziel_idx, layout = insert_special_node(g, layout, skel, ziel_coord)
        path = g.get_shortest_paths(start_idx, to=ziel_idx, weights="weight", output="vpath")[0]

        # path = path[::-1]        
        turns: list[Direction] = get_turn_sequence(layout, path)
        print(turns)
        
        out = {}
        for i, turn in enumerate(turns):
            out[f'Punkt_{i+1}'] = turn

        return turns
        # json_obj = json.dumps(out)

        # with open('Aufgabe2.json', 'w') as f:
        #     f.write(json_obj)


    return []
