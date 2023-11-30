from utils.ps_utils import WordVariable, RelationVariable, ExcludeProgram

class VersionSpace:
    def __init__(self, tt, tf, ft, programs, mappings):
        # TT: predicted true, groundtruth true
        self.tt, self.tf, self.ft = tt, tf, ft
        self.mappings = mappings
        self.programs = programs
        self.parent = None
        self.children = []


def is_counter(vs_main, vs_dependent):
    if not vs_dependent.tt - vs_main.tt:    # The true positive of two set must be disjoint
        return False
    if len(vs_dependent.tf - vs_main.tt) == 0: # The false positive of dependent must be a subset of main's true positive
        # TODO: support partial overlap in the future as well
        return False
    return True

def construct_counter_program(main_program, dependent_program):
    return ExcludeProgram(dependent_program, [main_program])

def join_version_space_counter(vs_main, vs_dependent):
    # Towards the goal of improving precision and recall
    tt_ = vs_dependent.tt
    tf_ = vs_dependent.tf - vs_main.tt
    ft_ = vs_dependent.ft
    wr = vs_dependent.programs[0].return_variables[0]
    out_mappings = set()
    for i, (w_bind, r_bind) in vs_dependent.mappings:
        if w_bind[wr] in vs_main.tt:
            continue
        out_mappings.add((i, (w_bind, r_bind)))

    return VersionSpace(tt_, tf_, ft_, construct_counter_program(vs_main.programs[0], vs_dependent.programs[0]), out_mappings)


def is_joinable(vs1, vs2):
    # Towards the goal of improving precision and recall
    # The two version spaces are joinable when
    # 1. one positive completely cover the other's negative, and after joining, F1 Score++
    # Towards this case one of the vs's precision will increase
    # 2. when the set of covered positive 
    pass


def join_version_space(vs1, vs2):
    # Towards the goal of improving precision and recall
    # The two version spaces are joinable when
    # 1. one positive completely cover the other's negative, and after joining, F1 Score++
    v1tf = vs1.tf
    v2tt = vs2.tt
    v2ft = vs2.ft
    excluded = v1tf - v2tt - v2ft
    # 2. when the set of covered positive 
    pass
