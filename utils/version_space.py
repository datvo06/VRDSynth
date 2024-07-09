from utils.ps_utils import (
        FalseValue, RelationConstraint, RelationLabelConstant, WordVariable, RelationVariable, ExcludeProgram, 
        FilterStrategy, AndConstraint, FindProgram, Hole, TrueValue, Constraint, LiteralSet, 
        WordLabelProperty, RelationLabelProperty,
        LabelEqualConstraint, RelationLabelEqualConstraint,
        UnionProgram,
        fill_hole)
from collections import defaultdict
from utils.misc import tuple2mapping, mapping2tuple
from typing import Set
import copy
from functools import lru_cache

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

class WordAndRelInBoundFilter(FilterStrategy):
    def __init__(self, find_program):
        self.word_set = find_program.word_variables
        self.rel_set = find_program.relation_variables
    
    def check_valid(self, program):
        if isinstance(program, WordVariable):
            return program in self.word_set
        if isinstance(program, RelationVariable):
            return program in self.rel_set
        return True
    def __hash__(self) -> int:
        return hash((self.word_set, self.rel_set))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, WordAndRelInBoundFilter):
            return False
        return self.word_set == o.word_set and self.rel_set == o.rel_set



class WordInBoundFilter(FilterStrategy):
    def __init__(self, find_program):
        self.word_set = find_program.word_variables
    
    def check_valid(self, program):
        if isinstance(program, WordVariable):
            return program in self.word_set
        return True

    def __hash__(self) -> int:
        return hash((self.word_set))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, WordInBoundFilter):
            return False
        return self.word_set == o.word_set



class NoDuplicateConstraintFilter(FilterStrategy):
    def __init__(self, constraint):
        self.constraint_set = set(self.gather_all_constraint(constraint))

    def gather_all_constraint(self, constraint):
        if isinstance(constraint, AndConstraint):
            lhs_constraints = self.gather_all_constraint(constraint.lhs)
            rhs_constraints = self.gather_all_constraint(constraint.rhs)
            return lhs_constraints + rhs_constraints
        else:
            return [constraint]

    def check_valid(self, program):
        if isinstance(program, Constraint):
            return program not in self.constraint_set
        return True

    def __hash__(self) -> int:
        return hash(self.constraint_set)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, NoDuplicateConstraintFilter):
            return False
        return self.constraint_set == o.constraint_set



class NoDuplicateLabelConstraintFilter(FilterStrategy):
    def __init__(self, constraint):
        self.constraint_set = set(NoDuplicateLabelConstraintFilter.gather_all_constraint(constraint))
        self.word_label = set()
        for constraint in self.constraint_set:
            if isinstance(constraint, LabelEqualConstraint):
                if isinstance(constraint.lhs, WordLabelProperty):
                    self.word_label.add(constraint.lhs.word_variable)
                elif isinstance(constraint.rhs, WordLabelProperty):
                    self.word_label.add(constraint.rhs.word_variable)
        self.rel_label = set()
        for constraint in self.constraint_set:
            if isinstance(constraint, RelationLabelEqualConstraint):
                if isinstance(constraint.lhs, RelationLabelProperty):
                    self.rel_label.add(constraint.lhs.relation_variable)
                elif isinstance(constraint.rhs, RelationLabelProperty):
                    self.rel_label.add(constraint.rhs.relation_variable)

    @staticmethod
    @lru_cache(maxsize=None)
    def gather_all_constraint(constraint):
        if isinstance(constraint, AndConstraint):
            lhs_constraints = NoDuplicateLabelConstraintFilter.gather_all_constraint(constraint.lhs)
            rhs_constraints = NoDuplicateLabelConstraintFilter.gather_all_constraint(constraint.rhs)
            return lhs_constraints + rhs_constraints
        else:
            return [constraint]

    def check_valid(self, program):
        if isinstance(program, WordLabelProperty):
            return program.word_variable not in self.word_label
        if isinstance(program, RelationLabelProperty):
            return program.relation_variable not in self.rel_label
        return True

    def __hash__(self) -> int:
        return hash(self.constraint_set)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, NoDuplicateConstraintFilter):
            return False
        return self.constraint_set == o.constraint_set


class DistinguishPropertyFilter(FilterStrategy):
    def __init__(self, mappings):
        pass

class RemoveFilteredConstraint(FilterStrategy):
    def __init__(self, constraint_set):
        self.constraint_set = constraint_set


class CompositeFilter(FilterStrategy):
    def __init__(self, filters):
        self.filters = filters

    def check_valid(self, program):
        for filter in self.filters:
            if not filter.check_valid(program):
                return False
        return True



def get_valid_cand_find_program(version_space: VersionSpace, program: FindProgram):
    if program.type_name() in LiteralSet:
        return []
    hole = Hole(Constraint)
    filterer = CompositeFilter([WordAndRelInBoundFilter(program), NoDuplicateConstraintFilter(program.constraint), NoDuplicateLabelConstraintFilter(program.constraint)])
    candidates = fill_hole(hole, 4, filterer)
    args = program.get_args()
    out_cands = []
    for cand in candidates:
        if isinstance(cand, TrueValue) or isinstance(cand, FalseValue):
            continue
        out_cands.append(cand)
    return out_cands


class NoDuplicateRelationConstraintFilter(FilterStrategy):
    def __init__(self, relation_constraint):
        self.rel_set = set([(r.w1, r.w2) for r in relation_constraint])

    def check_valid(self, program):
        if isinstance(program, RelationConstraint):
            # print(program.w1, program.w2, (program.w1, program.w2) in self.rel_set)
            # TODO: Allow this happen in the future
            return (program.w1, program.w2) not in self.rel_set
        return True

    def __hash__(self) -> int:
        return hash(self.rel_set)


class NoSelfRelationFilter(FilterStrategy):
    def __init__(self):
        super().__init__()

    def check_valid(self, program):
        if isinstance(program, RelationConstraint):
            return program.w1 != program.w2
        return True

class FixedRelationVarFilter(FilterStrategy):
    def __init__(self, relation_vars):
        self.relation_vars = relation_vars

    def check_valid(self, program):
        if isinstance(program, RelationVariable):
            return program in self.relation_vars
        return True

    def __hash__(self) -> int:
        return hash(self.relation_vars)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, FixedRelationVarFilter):
            return False
        return self.relation_vars == o.relation_vars


def get_valid_rel_constraint_program(version_space: VersionSpace, program: FindProgram):
    if program.type_name() in LiteralSet:
        return []
    # We have to add simultaneously two thing:
    # (1) A new relation variable and 
    # (2) A new relation constraint.
    rs = program.relation_variables
    new_r = len(rs) + 1
    new_r_var = RelationVariable(f"r{new_r}")
    # Along with this, choose all possible constraints
    hole = Hole(RelationConstraint)
    filterer = CompositeFilter([NoDuplicateRelationConstraintFilter(program.relation_constraint), WordInBoundFilter(program), NoSelfRelationFilter(), FixedRelationVarFilter([new_r_var])])
    candidates = fill_hole(hole, 4, filterer)
    return new_r_var, candidates


def construct_constraints_to_valid_version_spaces(vss):
    c2vs = defaultdict(set)
    for i, vs in enumerate(vss):
        for p in vs.programs:
            cs = get_valid_cand_find_program(vs, p)
            for c in cs:
                c2vs[c].add(i)
    return c2vs


def construct_rels_to_valid_version_spaces(vss):
    c2vs = defaultdict(set)
    for i, vs in enumerate(vss):
        for p in vs.programs:
            new_r_var, cs = get_valid_rel_constraint_program(vs, p)
            for c in cs:
                c2vs[new_r_var, c].add(i)
    return c2vs



def add_constraint_to_find_program(find_program, constraint):
    args = find_program.get_args()[:]
    args = copy.deepcopy(args)
    args[3] = AndConstraint(args[3], constraint)
    return FindProgram(*args)

def add_rel_to_find_program(find_program, rel_var, constraint):
    args = find_program.get_args()[:]
    args = copy.deepcopy(args)
    args[1].append(rel_var)
    args[2] = args[2] + [constraint]
    return FindProgram(*args)


def get_intersect_constraint_vs(c, vs, data_sample_set_relation_cache, cache) -> Set:
    vs_matches = set()
    for i, (w_bind, r_bind) in vs.mappings:
        nx_g = data_sample_set_relation_cache[i]
        if (i, (w_bind, r_bind)) in cache:
            if cache[(i, (w_bind, r_bind))]:
                vs_matches.add((i, (w_bind, r_bind)))
        else:
            w_bind, r_bind = tuple2mapping((w_bind, r_bind))
            val = c.evaluate(w_bind, r_bind, nx_g)
            if val:
                cache[(i, mapping2tuple((w_bind, r_bind)))] = True
                vs_matches.add((i, mapping2tuple((w_bind, r_bind))))
            else:
                cache[(i, mapping2tuple((w_bind, r_bind)))] = False
    return vs_matches


def get_intersect_rel_vs(rc: RelationConstraint, vs, data_sample_set_relation_cache, cache) -> Set:
    vs_matches = set()
    for i, (w_bind, r_bind) in vs.mappings:
        nx_g = data_sample_set_relation_cache[i]
        if (i, (w_bind, r_bind)) in cache:
            if cache[(i, (w_bind, r_bind))]:
                vs_matches.add((i, (w_bind, r_bind)))
        else:
            w_bind, r_bind = tuple2mapping((w_bind, r_bind))
            r_bind_old = r_bind
            r_bind = copy.deepcopy(r_bind)
            val = rc.evaluate(w_bind, nx_g)
            # add rbind to the mapping
            r_bind[rc.r] = (w_bind[rc.w1], w_bind[rc.w2], 0)
            if val:
                cache[(i, mapping2tuple((w_bind, r_bind_old)))] = True
                vs_matches.add((i, mapping2tuple((w_bind, r_bind))))
            else:
                cache[(i, mapping2tuple((w_bind, r_bind)))] = False
    return vs_matches


def join_counter_vss(pps, pcps, covered_tt_perfect, new_vss, covered_tt_counter):
    u_pcs = UnionProgram(pcps[:])
    u_pps = UnionProgram(pps[:])
    u_aps = UnionProgram(pcps[:] + pps[:])
    extra_pps = []
    extra_covered_tt = set()
    target_covered_tt = set((x[0], x[-1]) for x in covered_tt_perfect)
    for vs in new_vss:
        vs_tf = vs.tf - covered_tt_counter
        use_counter_program = False
        if len(vs_tf) < len(vs.tf):
            use_counter_program = True
        target_vs_tt = set((x[0], x[-1]) for x in vs.tt)
        target_vs_tf = set((x[0], x[-1]) for x in vs_tf)
        if not vs.tt - covered_tt_perfect - extra_covered_tt:
            continue
        if not target_vs_tf:
            extra_covered_tt |= vs.tt
            extra_pps.append(
                construct_counter_program(
                    u_pcs, vs.programs[0],
                )
            )
            continue

        if target_vs_tf - target_covered_tt: continue
        if not (target_vs_tt - target_covered_tt): continue
        new_vs_tt = target_vs_tt - target_covered_tt
        new_vs_tt = set([(x[0], x[1], x[2]) for x in vs.tt if x[2] not in new_vs_tt])
        extra_covered_tt |= new_vs_tt
        pps.append(
                construct_counter_program(
                    u_aps if use_counter_program else u_pps,
                    vs.programs[0],
                )
        )
    return extra_pps, extra_covered_tt


