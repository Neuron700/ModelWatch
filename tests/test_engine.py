import tempfile
import unittest
from pathlib import Path
import pandas as pd
from modelwatch.engine import analyze, evaluate, profile, read_csv, serialize
from modelwatch.demo import datasets
from modelwatch.store import save_run, history, clear

class Tests(unittest.TestCase):
    def test_ingestion(self):
        self.assertEqual(len(read_csv(b'a,b\n1,2\n')), 1)
        for data in (b'', b'a,a\n1,2', b'a,b\n'):
            with self.assertRaises(ValueError): read_csv(data)
    def test_quality_and_schema(self):
        a = pd.DataFrame({'x':[1,None,1], 'z':['a','a','a']})
        self.assertEqual(profile(a)['features'][0]['missing'], 1)
        r = analyze(a, pd.DataFrame({'x':[2,3,4], 'new':[1,2,3]}), ['x','z'])
        self.assertFalse(r['coverage_complete']); self.assertEqual(r['missing_columns'], ['z'])
    def test_demo_and_serialization(self):
        a,b = datasets(); r = analyze(a,b,['age','channel'])
        self.assertEqual(r['analyzed'], 2); self.assertGreater(r['drifted'], 0); serialize(r)
    def test_metrics(self):
        d = pd.DataFrame({'a':[0,1,None], 'p':[0,0,1]})
        r = evaluate(d,'a','p','classification')
        self.assertEqual(r['accuracy'], .5); self.assertEqual(r['rows_excluded'], 1)
        self.assertEqual(evaluate(pd.DataFrame({'a':[1.,2.], 'p':[2.,3.]}),'a','p','regression')['rmse'], 1)
        with self.assertRaises(ValueError): evaluate(d,'a','a','classification')
    def test_history(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d)/'db.sqlite'); save_run({'x':1}, p)
            self.assertEqual(history(p)[0][2], {'x':1}); clear(p); self.assertEqual(history(p), [])

if __name__ == '__main__': unittest.main()
