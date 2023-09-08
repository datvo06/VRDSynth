import re

import unidecode

from . import keywords

month_pattern = re.compile("(?:%s)[a-z]*" % ('|'.join(m.lower() for m in keywords.english_mounths)))
prefix = [
    'Part',
]
levels = [
    {
        "level": "I",
        "pattern": re.compile("(?i:%s){0,1}\s*[IVX]{1,3}\s*[\.\,\:\)]\s*.*" % ('|'.join(prefix))),
    },
    {
        "level": "A",
        "pattern": re.compile("(?i:%s){0,1}\s*[A-Z]{1}\s*[\.\,\:\)]\s*.*" % ('|'.join(prefix))),
    },
    {
        "level": "1",
        "pattern": re.compile("(?i:%s){0,1}\s*\d{1,2}\s*[\.\,\:\)]\s*.*" % ('|'.join(prefix))),
    },
    {
        "level": "a",
        "pattern": re.compile("(?i:%s){0,1}\s*[a-z]{1}\s*[\.\,\:\)]\s*.*" % ('|'.join(prefix))),
    },
]

priority = {}

for pre in prefix:
    for l in levels:
        priority['{0}_{1}'.format(pre, l['level'])] = len(priority)
    for l in levels:
        priority['{0}'.format(l['level'])] = len(priority)


def get_level(text):
    text_unicode = unidecode.unidecode(text).strip()
    if len(text) == 0:
        return None
    for level in levels:
        if level['pattern'].match(text_unicode):
            for pre in prefix:
                if pre.lower() in text_unicode.lower().replace(' ', '')[:len(pre) + 1]:
                    return '{0}_{1}'.format(pre, level['level'])
            return level['level']
    return get_bullet(text)


def get_bullet(text):
    if len(text.strip()) == 0:
        return None
    if text.split()[0] in ['o', '-', '+', '']:
        return 'bullet'
    text = text.strip().replace(' ', '')
    bullet = '•●*'
    if len(text) > 0:
        if text[0] in bullet and (len(text) == 1 or text[1] != text[0]):
            return 'bullet'
        elif text[0] in '-+':
            return text[0]
        else:
            return None
    else:
        return None


def is_title(text):
    text = text.strip()
    if text == '':
        return False
    if re.fullmatch('(?:\d{1,2}|[XIV]+)\s*[\.\:\.\/\,\)]{1}\s*[^\d].*', text) is not None:
        return True
    if get_level(text) not in [None, 'bullet', '+', '-']:
        return True
    if text.endswith(':') and len(text) < 30:
        return True
    return False


def is_time(text):
    if len(text) > 20:
        return False
    score = 0
    text = text.lower()
    mounts = month_pattern.findall(text)
    if len(mounts) > 0:
        score += 2
        for m in mounts:
            text = text.replace(m, ' ')
    quarter = re.findall('(?:$|[^\d]{1})\d{4}(?:$|[^\d]{1})', text)
    if len(quarter) > 0:
        score += 4
        for q in quarter:
            text = text.replace(q, ' ')
    double = re.findall('(?:$|[^\d]{1})\d{1,2}(?:$|[^\d]{1})', text)
    if len(double) > 0:
        for d in double:
            text = text.replace(d, ' ')
        score += 2
    if score >= 4 and len(text.replace(' ', '')) < 10:
        return True
    else:
        return False


def can_merge_text(text1, text2):
    text1 = text1.strip()
    text2 = text2.strip()
    if text1 == '' or text2 == '':
        return False

    if text1[-1] in ['-', '&']:
        return True
    elif len(unidecode.unidecode(text1).split()) == 0 or unidecode.unidecode(text1).split()[-1].lower() in ['and']:
        return True
    elif is_time(text1) and is_time(text2) and (text1.endswith('-') or text2.startswith('-')):
        return True
    return False


class Span:
    def __init__(self, start, end, idx=0):
        self.start = start
        self.end = end
        self.idx = idx

    def __repr__(self):
        return "Span({0} {1})".format(self.start, self.end)


def findall(text, sub):
    return [i.start() for i in re.finditer(re.escape(sub), text)]


def fuzzy_match(text, sub):
    spans = findall(text, sub)
    spans = [Span(i, i + len(sub)) for i in spans]
    if len(spans) == 0:
        bounding_boxes = []
        tokens = sub.split()
        box_idx = 0
        for itoken, token in enumerate(tokens):
            if len(token) >= 2 or itoken == 0 or itoken == len(tokens) - 1:
                r1 = findall(text, token)
                r_intersect = [Span(i, i + len(token)) for i in r1]
                if len(r_intersect) > 0:
                    for b in r_intersect:
                        b.idx = box_idx
                        box_idx += 1
                    bounding_boxes.append(r_intersect)
        if len(bounding_boxes) > 0:
            sequences = []
            max_len = 0
            for box in bounding_boxes[0]:
                sequence = [box]
                for after_boxes in bounding_boxes[1:]:
                    for after_box in after_boxes:
                        if after_box.start > box.start:
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
                    spans = [Span(seq[0].start, seq[-1].end) for seq in sequences]
    return spans
