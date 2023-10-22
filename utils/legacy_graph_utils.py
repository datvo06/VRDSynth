#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: graph_utils.py
ultilities for graph
"""
import numpy as np
import re
from collections import defaultdict
import networkx as nx
import os
import cv2


def check_intersect_range(x1, l1, x2, l2):
    if x1 > x2:
        x1, x2 = x2, x1
        l1, l2 = l2, l1
    return (x1+l1) > x2


def get_intersect_range(x1, l1, x2, l2):
    if x1 > x2:
        x1, x2 = x2, x1
        l1, l2 = l2, l1
    if not check_intersect_range(x1, l1, x2, l2):
        return 0
    if (x1 + l1) > (x2+l2):
        return l2
    else:
        return x1 + l1 - x2


def is_horz_intersect(bbox1, bbox2):
    return check_intersect_range(bbox1[1], bbox1[3], bbox2[1], bbox2[3])


def is_vert_intersect(bbox1, bbox2):
    return check_intersect_range(bbox1[0], bbox1[2], bbox2[0], bbox2[2])


def get_intersect_range_horizontal_proj(bbox1, bbox2):
    return get_intersect_range(bbox1[1], bbox1[3], bbox2[1], bbox2[3])


def get_intersect_range_vertical_proj(bbox1, bbox2):
    return get_intersect_range(bbox1[0], bbox1[2], bbox2[0], bbox2[2])


class Entity(object):

    threshhold_really_horizontal = 2.0
    threshold_really_vertical = 0.2

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.lefts, self.rights, self.tops, self.bottoms = [], [], [], []

    def get_bb(self):
        return (self.x, self.y, self.w, self.h)


    def is_left_of(self, oc, ref_cells):
        """Check if this cell is directly left of
        other cell given a set of full cells sorted"""
        # 0. Check if oc in self.rights
        if oc in self.rights: return True

        if oc.x < self.x or not is_horz_intersect(
            self.get_bb(), oc.get_bb()
        ):
            return False
        if get_intersect_range_horizontal_proj(
            self.get_bb(), oc.get_bb()
        ) > 0.9*min(self.h, oc.h):
            if oc.x - self.x < 0.1*min(self.w, oc.w):
                return True

        # => right now: other cell is on the right and intersect on projection
        # horizontal side

        if len(ref_cells) == 0:
            return True

        # 1. get all cells on the right of this cell.
        # meaning all cells that have overlapping regions with this cell
        # and lie to the right
        ref_cells = [cell for cell in ref_cells
                     if is_horz_intersect(
                         self.get_bb(), cell.get_bb()) and
                     (cell.x + cell.w) < oc.x + oc.w*0.1 and
                     cell.x >= (self.x+self.w*0.8) and
                     is_horz_intersect(
                         self.get_bb(), cell.get_bb())
                     ]
        # 2. filters all the small overlapping cells
        ref_cells = [cell for cell in ref_cells
                     if get_intersect_range_horizontal_proj(
                         self.get_bb(),
                         cell.get_bb()
                     ) > min(self.h, cell.h) / 5]
        ref_cells = [cell for cell in ref_cells
                     if
                     get_intersect_range_horizontal_proj(
                         cell.get_bb(),
                         oc.get_bb()
                     ) > oc.h / 2 or
                     get_intersect_range_horizontal_proj(
                         self.get_bb(),
                         cell.get_bb()
                     ) > min(cell.h, self.h)*0.8
                     ]

        # 3. Check if there are any cells lies between this and oc
        if len(ref_cells) > 0:
            return False

        # 4. return results
        return True

    def is_right_of(self, oc, ref_cells):
        return oc.is_left_of(self, ref_cells)

    def is_top_of(self, oc, ref_cells):
        """Check if this cell is directly top of
        other cell given a set of full cells sorted"""
        # 0. Check if oc in self.rights
        if oc in self.bottoms:
            return True

        if oc.y < self.y or not is_vert_intersect(
                self.get_bb(), oc.get_bb()):
            return False

        if get_intersect_range_vertical_proj(
            self.get_bb(), oc.get_bb()) < min(
                self.w, oc.w)/5:
            return False

        if not ref_cells:
            return True

        ref_cells = [cell for cell in ref_cells
                     if is_vert_intersect(
                         self.get_bb(), cell.get_bb()) and
                     (cell.y + cell.h) < oc.y + oc.h*0.1 and
                     cell.y >= (self.y+self.h*0.8) and
                     is_vert_intersect(
                         self.get_bb(), cell.get_bb())
                     ]
        # 2. filters all the small overlapping cells
        ref_cells = [cell for cell in ref_cells
                     if
                     get_intersect_range_vertical_proj(
                         self.get_bb(),
                         cell.get_bb()
                     ) > min(self.w, cell.w) / 5]
        ref_cells = [cell for cell in ref_cells
                     if get_intersect_range_vertical_proj(
                         cell.get_bb(),
                         oc.get_bb()
                     ) > oc.w / 2 or
                     get_intersect_range_vertical_proj(self.get_bb(),
                                                       cell.get_bb()
                                                       ) >
                     min(self.w, cell.w)*0.8
                     ]

        return not ref_cells


    def __getitem__(self, key):
        return self.get_bb()[key]


def _get_v_intersec(loc1, loc2):
    _, y11, _, h1 = loc1
    _, y21, _, h2 = loc2
    y12 = y11 + h1
    y22 = y21 + h2
    ret = max(0, min(y12 - y21, y22 - y11))
    return ret


def _get_v_union(loc1, loc2):
    _, y11, _, h1 = loc1
    _, y21, _, h2 = loc2
    y12 = y11 + h1
    y22 = y21 + h2
    ret = min(h1 + h2, max(y22 - y11, y12 - y21))
    return ret


def _get_h_intersec(loc1, loc2):
    x11, y11, w1, h1 = loc1
    x21, y21, w2, h2 = loc2
    x12 = x11 + w1
    x22 = x21 + w2
    ret = max(0, min(x12 - x21, x22 - x11))
    return ret


def _get_h_union(loc1, loc2):
    x11, y11, w1, h1 = loc1
    x21, y21, w2, h2 = loc2
    x12 = x11 + w1
    x22 = x21 + w2
    ret = min(w1 + w2, max(x22 - x11, x12 - x21))
    return ret


def get_nearest_line(cr_line, list_lines, dr='l', thresh=50000):
    line_loc = cr_line.get_bb()
    ret = None
    dt = thresh
    for line in list_lines:
        loc = line.get_bb()
        if dr in {'r', 'l'}:
            if _get_v_intersec(loc, line_loc) <= 0.3 * _get_v_union(loc, line_loc):
                continue
        elif dr in {'t', 'b'}:
            d = min(abs(loc[1] - line_loc[1] - line_loc[3]),
                    abs(line_loc[1] - loc[1] - loc[3]))

            # 0.1 * _get_h_union(loc, line_loc):
            if _get_h_intersec(loc, line_loc) <= 0:
                if dr == 't' and line_loc[1] > loc[1]:
                    continue
                if not (dr == 't' and d < 0.5 * line_loc[3]
                        and line_loc[1] + 1.3 * line_loc[3] > loc[1]):
                    continue

        dist = dt + 1
        if dr == 'r':
            if loc[0] > line_loc[0]:
                dist = loc[0] - line_loc[0] - line_loc[2]
        elif dr == 'l':
            if loc[0] < line_loc[0]:
                dist = line_loc[0] - loc[0] - loc[2]
        elif dr == 'b':
            if loc[1] > line_loc[1]:
                dist = loc[1] - line_loc[1] - line_loc[3]
        elif dr == 't':
            if loc[1] < line_loc[1]:
                dist = line_loc[1] - loc[1] - loc[3]
        if dist < dt:
            ret = line
            dt = dist
    return ret


class Graph():
    edge_labels = [
        'lr',
        'rl',
        'tb',
        'bt',
    ]

    def __init__(self, locs):
        self.org_items = list([Entity(*loc) for loc in locs])
        self.nodes = self.org_items
        self.es = []
        self.build_edges()
        self._get_adj_matrix()
        print(len(self.es))

    def build_edges(self):
        cell_list_top_down = sorted(self.nodes, key=lambda cell: cell.y)
        cell_list_left_right = sorted(self.nodes, key=lambda cell: cell.x)
           # 1.1 Check this cell with every cell to the right of it
           # TODO: More effective iteration algo e.g: cached collisions matrix
        self._build_lr_edges(cell_list_top_down)
           # 2. top-down
        self._build_td_edges_1(cell_list_left_right)
        # clean left-right edges
        self._clean_lr_edges()
        # clean top-bot edges
        self._clean_td_edges()

    def _build_lr_edges(self, td_cells):
        for cell in td_cells:
            cell_collide = [oc
                            for oc in td_cells 
                            if oc.x >= cell.x and
                            is_horz_intersect(
                                cell.get_bb(), oc.get_bb()
                            ) and cell != oc]
            cell_collide = [oc
                            for oc in cell_collide
                            if get_intersect_range_horizontal_proj(
                                cell.get_bb(), oc.get_bb()
                            ) > min(cell.h, oc.h)*0.4]

            for oc in cell_collide:
                if cell.is_left_of(oc, cell_collide
                                   ) and oc not in cell.rights:
                    self.es.append(
                        (cell, oc, self.edge_labels.index('lr')))
                    self.es.append(
                        (oc, cell, self.edge_labels.index('rl')))
                    cell.rights.append(oc)
                    oc.lefts.append(cell)

    def _clean_lr_edges(self):
        for cell in self.nodes:
            if len(cell.lefts) <= 1:
                continue
            lcells = sorted(cell.lefts, key=lambda x: x.x)
            removes = [c for c in lcells if c.x + c.w >
                       cell.x and c.x > cell.x - 0.5 * cell.h]
            lcells = list(set(lcells) - set(removes))
            # cluster these cell into column:

            columns = []
            column_cells = []
            # column_x = lcells[0].x
            for c in lcells:
                its = 0
                union = 100
                if column_cells:
                    its = get_intersect_range_vertical_proj(
                        column_cells[-1].get_bb(), c.get_bb())
                    union = min(column_cells[-1].w, c.w)
                if its > 0.5 * union:
                    column_cells.append(c)
                    continue
                else:
                    if column_cells:
                        columns.append(column_cells)
                    column_cells = [c]
                    # column_x = c.x
            if column_cells:
                columns.append(column_cells)

            # lcells to keep:
            if len(columns) > 0:
                real_lefts = columns[-1]
            else:
                real_lefts = []
            removes += [c for c in lcells if c not in real_lefts]
            remove_edges = []
            for c in removes:
                c.rights.remove(cell)
                for i, j, lbl in self.es:
                    if i == c and j == cell and lbl == self.edge_labels.index('lr'):
                        remove_edges.append((i, j, lbl))
                    if i == cell and j == c and lbl == self.edge_labels.index('rl'):
                        remove_edges.append((i, j, lbl))
            [self.es.remove(e) for e in remove_edges]

            cell.lefts = real_lefts

    def _build_td_edges(self, lr_cells):
        for cell in lr_cells:
            cell_collide = [
                    oc for oc in lr_cells
                    if oc.y > cell.y + cell.h * 0.6 and
                    is_vert_intersect(
                        cell.get_bb(), oc.get_bb()
                    ) and cell != oc]
            for oc in cell_collide:
                if cell.is_top_of(oc, cell_collide) and\
                        oc not in cell.bottoms:
                    self.es.append(
                        (cell, oc, self.edge_labels.index('tb')))
                    self.es.append(
                        (oc, cell, self.edge_labels.index('bt')))
                    cell.bottoms.append(oc)
                    oc.tops.append(cell)

    def _build_td_edges_1(self, lr_cells):
        for c in lr_cells:
            top_cell = get_nearest_line(c, lr_cells, 't')
            if top_cell:
                self.es.append(
                    (top_cell, c, self.edge_labels.index('tb')))
                self.es.append(
                    (c, top_cell, self.edge_labels.index('bt')))
                c.tops.append(top_cell)
                top_cell.bottoms.append(c)

    def _clean_td_edges(self):
        for cell in self.nodes:
            if len(cell.tops) <= 1: continue
            top_cells = sorted(cell.tops, key=lambda x: x.y)

            rows, row_cells = [], []
            for c in top_cells:
                its = 0
                union = 10000
                if row_cells:
                    its = get_intersect_range_horizontal_proj(
                        row_cells[-1].get_bb(), c.get_bb())
                    union = min(row_cells[-1].w, c.w)
                if its > 0.5 * union:
                    row_cells.append(c)
                    continue
                else:
                    if len(row_cells) > 0:
                        rows.append(row_cells)
                    row_cells = [c]
            if row_cells:
                rows.append(row_cells)

            # lcells to keep:
            real_tops = rows[-1]
            rms = [c for c in top_cells if c not in real_tops]
            rm_es = []
            for c in rms:
                c.bottoms.remove(cell)
                for (i, j, lbl) in self.es:
                    if i == c and j == cell and lbl == self.edge_labels.index('tb'):
                        rm_es.append((i, j, lbl))
                    if i == cell and j == c and lbl == self.edge_labels.index('bt'):
                        rm_es.append((i, j, lbl))
            [self.es.remove(e) for e in rm_es]

            cell.tops = real_tops


    def _get_adj_matrix(self):
        def scale_coor(node):
            scale_x1 = (node.x - min_x) / max_delta_x
            scale_y1 = (node.y - min_y) / max_delta_y
            scale_x1b = (node.x + node.w - min_x) / max_delta_x
            scale_y1b = (node.y + node.h - min_y) / max_delta_y
            return scale_x1, scale_y1, scale_x1b, scale_y1b

        def dist(x1, y1, x2, y2):
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        def rect_distance(rect1, rect2):
            x1, y1, x1b, y1b = rect1
            x2, y2, x2b, y2b = rect2

            left = x2b < x1
            right = x1b < x2
            bottom = y2b < y1
            top = y1b < y2

            if top and left:
                return dist(x1, y1b, x2b, y2)
            elif left and bottom:
                return dist(x1, y1, x2b, y2b)
            elif bottom and right:
                return dist(x1b, y1, x2, y2b)
            elif right and top:
                return dist(x1b, y1b, x2, y2)
            elif left:
                return x1 - x2b
            elif right:
                return x2 - x1b
            elif bottom:
                return y1 - y2b
            elif top:
                return y2 - y1b

            return 0.


        adj = np.zeros((len(self.nodes), len(self.edge_labels), len(self.nodes)))

        max_x = np.max([n.x + n.w for n in self.nodes])
        max_y = np.max([n.y + n.h for n in self.nodes])
        min_x = np.min([n.x for n in self.nodes])
        min_y = np.min([n.y for n in self.nodes])

        max_delta_x = np.abs(max_x - min_x)
        max_delta_y = np.abs(max_y - min_y)
        self.adj = adj.astype(np.float16)


def from_funsd_datasample(data):
    customized_loc = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in data['boxes']]
    g = Graph(customized_loc)
    adj = g.adj
    # adj: (num_nodes, num_edge_labels, num_nodes)
    # Convert adj to sparse matrix by nonzeros
    list_edges = []
    for etype in range(adj.shape[1]):
        for i in range(adj.shape[0]):
            for j in range(adj.shape[2]):
                if adj[i, etype, j] > 0:
                    list_edges.append((i, j, etype))

    return list_edges


def build_nx_g_legacy(data):
    edges = from_funsd_datasample(data)
    # build a networkx graph
    nx_g = nx.MultiDiGraph()
    for i, j, etype in edges:
        # label is the index of max projection
        label = etype
        center_i = [(data['boxes'][i][0] + data['boxes'][i][2]) / 2,
                    (data['boxes'][i][1] + data['boxes'][i][3]) / 2]
        center_j = [(data['boxes'][j][0] + data['boxes'][j][2]) / 2,
                    (data['boxes'][j][1] + data['boxes'][j][3]) / 2]
        mag = np.sqrt((center_i[0] - center_j[0]) ** 2 + (center_i[1] - center_j[1]) ** 2)
        nx_g.add_edge(i, j, mag=mag, lbl=label)
    for i, (box, label, word) in enumerate(zip(data.boxes, data.labels, data.words)):
        if i not in nx_g.nodes():
            nx_g.add_node(i)
        nx_g.nodes[i].update({'x0': box[0], 'y0': box[1], 'x1': box[2], 'y1': box[3], 'label': label, 'word': word})
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



if __name__ == '__main__':
    from utils.funsd_utils import viz_data
    import argparse
    import pickle as pkl
    dataset = pkl.load(open('funsd_cache_word_merging_vrdsynth_dummy_0.5/dataset.pkl', 'rb'))
    os.makedirs('funsd_cache_word_merging_vrdsynth_dummy_0.5/viz_legacy', exist_ok=True)
    for i, data in enumerate(dataset):
        nx_g = build_nx_g_legacy(data)
        img = viz_data(data, nx_g)
        cv2.imwrite(f'funsd_cache_word_merging_vrdsynth_dummy_0.5/viz_legacy/{i}.png', img)
