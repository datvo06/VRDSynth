import re
from typing import *
import numpy as np
from file_reader.layout.box import group_by_row
from file_reader.layout.page import Page


def pad_image(image, width):
    img_height, img_width, channel = image.shape
    res = np.ones((img_height, width, channel), dtype='uint8') * 255
    pad_left = int(width - img_width) // 2
    res[:, pad_left:pad_left + img_width, :] = image
    return pad_left, res


def remove_header(pages: List[Page], verbose=False):
    page_lines = []
    counter = {}
    for i, page in enumerate(pages):
        height = page.height
        textlines = page.textlines
        if len(textlines) == 0:
            continue
        lines = group_by_row(textlines)
        page_lines.append(lines)
        if len(lines) > 0 and lines[0][0].y0 < 0.15 * height:
            first_line_origin = ' '.join(map(lambda x: x.text.strip(), lines[0]))
            first_line = re.sub('\d+', '0', first_line_origin)
            first_line = re.sub('\s+', ' ', first_line).strip()
            if verbose:
                print('first line:', first_line_origin)
            if first_line in counter:
                counter[first_line] += 1
            else:

                counter[first_line] = 1
            if counter[first_line] > 1:
                page.header = lines[0]
                lines = lines[1:]
                if verbose:
                    print('header:', first_line_origin)
            else:
                page.header = []

        textlines = []
        for line in lines:
            textlines.extend(line)
        page.textlines = textlines
    return pages


def remove_footer(pages: List[Page], verbose=False):
    page_lines = []
    counter = {}
    last_lines = []
    for i, page in enumerate(pages):
        height = page.height
        textlines = page.textlines
        if len(textlines) == 0:
            page_lines.append([])
            last_lines.append((None, None))
            continue

        lines = group_by_row(textlines)
        page_lines.append(lines)
        if len(lines) > 0 and lines[-1][0].y1 > 0.85 * height:
            last_line_origin = ' '.join(map(lambda x: x.text.strip(), lines[-1]))
            last_line = re.sub('\d+', '0', last_line_origin)
            last_line = re.sub('\s+', ' ', last_line).strip()
            last_lines.append((last_line, last_line_origin))
            if last_line in counter:
                counter[last_line] += 1
            else:
                counter[last_line] = 1
        else:
            last_lines.append((None, None))

    for page, lines, (last_line, last_line_origin) in zip(pages, page_lines, last_lines):
        if verbose:
            print('last line:', last_line_origin)
        if (last_line in counter and counter[last_line] > 1):
            page.footer = lines[-1]
            lines = lines[:-1]
            if verbose:
                print('footer:', last_line_origin)
        else:
            page.footer = []
        textlines = []
        for line in lines:
            textlines.extend(line)
        page.textlines = textlines
    return pages
