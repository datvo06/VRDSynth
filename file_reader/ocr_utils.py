def check_in_group_collected(block, group_collected):
    for group_info in group_collected:
        if check_in_subgroup(block, group_info):
            return True
    return False

def check_in_subgroup(block, subgroup):
    top = subgroup['x1']
    bottom = subgroup['y1']
    block_center = (block['x1'] + block['y1']) / 2
    if block_center >= subgroup['x1'] and block_center <= subgroup['y1']:
        return True
    return False