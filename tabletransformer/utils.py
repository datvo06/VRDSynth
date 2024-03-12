from utils.data_sample import DataSample, Bbox
import networkx as nx
import glob
from utils.misc import pexists, pjoin
import os.path
from utils.legacy_graph_utils import build_nx_g_legacy, from_funsd_datasample
import json
import copy
from tabletransformer.inference import iob
from collections import defaultdict
import numpy as np

def build_nx_g_legacy_table_cells(entity_data: DataSample, table_out_dir: str):
    """
    """
    # First, load all the table
    img_fp = entity_data.img_fp
    if not pexists(img_fp):
        img_fp = img_fp.replace(".jpg", ".png")
    img_fname = os.path.basename(img_fp)
    ext = os.path.splitext(img_fname)[1]
    cell_fp = pjoin(table_out_dir, img_fname.replace(ext, "_rev_cells.json"))
    if not cell_fp:    # There are no table
        return build_nx_g_legacy(entity_data)
    tabs = json.load(open(cell_fp))
    assert isinstance(tabs, list)
    for tab in tabs:
        assert isinstance(tab, list)
        for cell in tab:
            assert isinstance(cell, dict)
            # Only check table row and table column


def build_nx_g_legacy_table_rc(entity_data: DataSample, table_out_dir: str):
    """
    """
    # First, load all the table
    img_fp = entity_data.img_fp
    if not pexists(img_fp):
        img_fp = img_fp.replace(".jpg", ".png")
    img_fname = os.path.basename(img_fp)
    ext = os.path.splitext(img_fname)[1]
    obj_fp = pjoin(table_out_dir, img_fname.replace(ext, "_rev_objects.json"))
    if not obj_fp:    # There are no table
        return build_nx_g_legacy(entity_data)
    tabs = json.load(open(obj_fp))
    assert isinstance(tabs, list)
    ir2es = defaultdict(list)
    ic2es = defaultdict(list)
    ir2bbox = {}
    ic2bbox = {}
    for tidx, tab in enumerate(tabs):
        assert isinstance(tab, list)
        tab_rs = [o for o in tab if o["label"] == "table row"]
        tab_cs = [o for o in tab if o["label"] == "table column"]
        for ridx, r in enumerate(tab_rs):
            ir2bbox[(tidx, ridx)] = Bbox(r["x0"], r["y0"], r["x1"], r["y1"])
        for cidx, c in enumerate(tab_cs):
            ic2bbox[(tidx, cidx)] = Bbox(c["x0"], c["y0"], c["x1"], c["y1"])
        for eidx, bbox in enumerate(entity_data.boxes):
            ers = [i for i, o in enumerate(tab_rs) if iob(bbox, Bbox(*o["bbox"])) >= 0.3]
            ecs = [i for i, o in enumerate(tab_cs) if iob(bbox, Bbox(*o["bbox"])) >= 0.3]
            for r in ers:
                ir2es[tidx, r].append(eidx)
            for c in ecs:
                ic2es[tidx, c].append(eidx)

    # Now, build the graph
    # Same row would be index 5
    # Same column would be index 6
    edges = from_funsd_datasample(entity_data)
    data = entity_data
    nx_g = nx.MultiDiGraph()
    for i, (box, label, word) in enumerate(zip(data.boxes, data.labels, data.words)):
        if i not in nx_g.nodes():
            nx_g.add_node(i)
        nx_g.nodes[i].update({'x0': box[0], 'y0': box[1], 'x1': box[2], 'y1': box[3], 'label': label, 'word': word})

    for (tidx, r), es in ir2es.items():
        # Create a new node for the row
        row_node = max(nx_g.nodes() if len(nx_g.nodes()) >= 0 else [-1]) + 1
        nx_g.add_node(row_node)
        nx_g.nodes[row_node].update({'label': "trow", 'x0': ir2bbox[(tidx, r)].x0, 'y0': ir2bbox[(tidx, r)].y0, 'x1': ir2bbox[(tidx, r)].x1, 'y1': ir2bbox[(tidx, r)].y1, 'word': ''})
        for i in range(len(es)):
            edges.append((row_node, es[i], 5))
            edges.append((es[i], row_node, 5))
    for (tidx, c), es in ic2es.items():
        # Create a new node for the column
        col_node = max(nx_g.nodes() if len(nx_g.nodes()) >= 0 else [-1]) + 1
        nx_g.add_node(col_node)
        nx_g.nodes[col_node].update({'label': "tcol", 'x0': ic2bbox[(tidx, c)].x0, 'y0': ic2bbox[(tidx, c)].y0, 'x1': ic2bbox[(tidx, c)].x1, 'y1': ic2bbox[(tidx, c)].y1, 'word': ''})

        for i in range(len(es)):
            edges.append((col_node, es[i], 6))
            edges.append((es[i], col_node, 6))

    for i, j, etype in edges:
        # label is the index of max projection
        label = etype
        center_i = [(data['boxes'][i][0] + data['boxes'][i][2]) / 2,
                    (data['boxes'][i][1] + data['boxes'][i][3]) / 2]
        center_j = [(data['boxes'][j][0] + data['boxes'][j][2]) / 2,
                    (data['boxes'][j][1] + data['boxes'][j][3]) / 2]
        mag = np.sqrt((center_i[0] - center_j[0]) ** 2 + (center_i[1] - center_j[1]) ** 2)
        nx_g.add_edge(i, j, mag=mag, lbl=label)

    # Normalize the mag according to the smallest and largest mag
    mags = [e[2]['mag'] for e in nx_g.edges(data=True)]
    if len(mags) > 1:
        min_mag = min(mags)
        max_mag = max(mags)
        for e in nx_g.edges(data=True):
            e[2]['mag'] = (e[2]['mag'] - min_mag) / (max_mag - min_mag)
    # normalize the coord according to the largest coord
    max_coord_x = max([e[1]['x1'] for e in nx_g.nodes(data=True)])
    max_coord_y = max([e[1]['y1'] for e in nx_g.nodes(data=True)])
    min_coord_x = min([e[1]['x0'] for e in nx_g.nodes(data=True)])
    min_coord_y = min([e[1]['y0'] for e in nx_g.nodes(data=True)])
    for _, n in nx_g.nodes(data=True):
        n['x0'] = (n['x0'] - min_coord_x) / (max_coord_x - min_coord_x)
        n['y0'] = (n['y0'] - min_coord_y) / (max_coord_y - min_coord_y)
        n['x1'] = (n['x1'] - min_coord_x) / (max_coord_x - min_coord_x)
        n['y1'] = (n['y1'] - min_coord_y) / (max_coord_y - min_coord_y)
    return nx_g
