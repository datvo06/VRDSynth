from file_reader.layout.box import Box, group_by_row, sort_box
from file_reader.layout.paragraph import Paragraph
import fitz


def add_annotation(page, textline, label=None, color=None, title=None, exceptions=[]):
    text = textline['text']
    l_x1 = textline['x1']
    l_x0 = textline['x0']
    l_y1 = textline['y1']
    l_y0 = textline['y0']
    except_boxs = []
    for ex in exceptions:
        r1 = page.search_for(ex, hit_max=20, quads=False, flags=1)
        except_boxs.extend([Box.from_rect(r) for r in r1])

    def is_except_box(_bbox):
        if len(except_boxs) == 0:
            return False
        return any(ex_box.intersect(_bbox) for ex_box in except_boxs)

    rect = Box(l_x0, l_y0, l_x1, l_y1)
    r1 = page.search_for(text, hit_max=20, quads=False, flags=1)

    r_intersect = [Box.from_rect(r) for r in r1 if rect.intersect(r) and not is_except_box(r)]
    if len(r_intersect) == 0:
        r_intersect = [Box.from_rect(r) for r in r1]

    rows = group_by_row(r_intersect)
    if len(rows) == 0 and len(text) >= 3:
        bounding_boxes = []
        tokens = text.split()
        box_idx = 0
        for itoken, token in enumerate(tokens):
            if len(token) >= 2 or itoken == 0 or itoken == len(tokens) - 1:
                r1 = page.search_for(token, hit_max=50, quads=False, flags=1)
                r_intersect = [Box.from_rect(r) for r in r1 if rect.intersect(r) and not is_except_box(r)]
                r_intersect = sort_box(r_intersect)
                if len(r_intersect) > 0:
                    for b in r_intersect:
                        b.idx = box_idx
                        box_idx += 1
                    bounding_boxes.append(r_intersect)
                else:
                    bounding_boxes.append([])

        if len(bounding_boxes) > 0:
            sequences = []
            max_len = 0
            for box in bounding_boxes[0]:
                sequence = [box]
                for after_boxes in bounding_boxes[1:]:
                    for after_box in after_boxes:
                        if after_box.is_after(box):
                            sequence.append(after_box)
                            box = after_box
                            break
                sequences.append(sequence)
                max_len = max(max_len, len(sequence))
            sequences = [seq for seq in sequences if len(seq) == max_len]
            if len(sequences) > 0:
                groups = []
                group = []
                gr = set()
                for seq in sequences:
                    s_seq = set(box.idx for box in seq)
                    if len(gr & s_seq) > 0:
                        group.append(seq)
                        gr = gr | s_seq
                    else:
                        if len(group) > 0:
                            groups.append(group)
                        group = [seq]
                        gr = s_seq
                if len(group) > 0:
                    groups.append(group)
                if len(groups) > 0:
                    sequences = [min(seqs, key=lambda seq: seq[-1].idx - seq[0].idx) for seqs in groups]
                    boxes = [b for seq in sequences for b in seq]
                    if len(boxes) == len(bounding_boxes):
                        rows = group_by_row(boxes)
    if len(rows) == 0:
        return False
    rects = []
    for row in rows:
        r = Paragraph(row)
        pad = 0.1 * r.height
        r = fitz.Rect(r.x0 + pad / 2, r.y0 + pad, r.x1 - pad / 2, r.y1 - pad)
        rects.append(r)
    annot = page.add_highlight_annot(rects)
    if color:
        annot.set_colors({'stroke': color, 'fill': color})
    annot.set_info({"content": label, "title": title})
    annot.update()
    return True
