"""Pure task/program/statistical fixtures: zero AGT model constructions or forwards."""
import unittest,random
import torch
from task_data_v1 import interpret, independent_label, candidate, validate_family, encode, below, permutation
from program_adapter_v1 import PROGRAMS, ResearchProgram
from analysis_v1 import quantile

class TaskTests(unittest.TestCase):
    def test_scalar_copy_and_alias(self):
        c=[['SET',0,None,1],['COPY',1,0,None],['SET',0,None,0],['QUERY',1,None,None],['COPY',1,1,None],['QUERY',1,None,None]]
        self.assertEqual(interpret(c)[1],[1,1]);self.assertEqual(independent_label(c),1)
    def test_illegal_fields(self):
        for c in ([['SET',8,None,0]],[['XOR',0,None,2]],[['COPY',0,True,None]],[['QUERY',0,None,1]]):
            with self.assertRaises(AssertionError):interpret(c)
    def test_deterministic_families(self):
        for L in (64,96):
            a=candidate(random.Random(6100),L);b=candidate(random.Random(6100),L)
            self.assertEqual(a,b);validate_family(*a)
    def test_query_encoding_has_no_answer(self):
        x=encode([['QUERY',3,None,None]])
        self.assertEqual(x.shape,(1,1,64));self.assertEqual(x.sum().item(),2)
        self.assertEqual(x[0,0,12:].count_nonzero().item(),0)
    def test_programs(self):
        expected=((0,1,2,3,4,5,6,7),(0,1,2,3,3,5,6,7),(0,1,2,4,4,5,6,7))
        for p,blocks in zip(PROGRAMS,expected):
            r=p.validate();self.assertEqual(tuple(x['block'] for x in r),blocks);self.assertEqual(r[-1]['execution_after'],8)
    def test_fail_closed_scripts(self):
        for actions in ((8,)*8,(0,)*8,tuple(range(7)),tuple(range(8))+(8,)):
            with self.assertRaises(ValueError):ResearchProgram('fixture',actions).validate()
    def test_quantile(self):
        self.assertEqual(quantile([0.,1.],.025),.025)
        self.assertEqual(quantile([0.,1.],.975),.975)
    def test_rng_bounds_and_permutation(self):
        r=random.Random(6100);self.assertEqual(below(r,1),0)
        self.assertEqual(sorted(permutation(r,range(8))),list(range(8)))

if __name__=='__main__':unittest.main(verbosity=2)
