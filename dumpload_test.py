import unittest
import nest
from dumpload import *


class TestSerializationMethods(unittest.TestCase):

    def setUp(self) -> None:
        nest.set_verbosity("M_ERROR")
        nest.ResetKernel()

    def test_dump(self):
        nest.ResetKernel()
        a = nest.Create("aeif_psc_alpha", 5)
        b = nest.Create("aeif_psc_alpha", 5)
        nest.Connect(a, b)
        thedump = Dump()
        onlysynapses = Dump(selections=["synapses"])
        onlynodes = Dump(selections=["nodes"])
        self.assertIsNotNone(thedump)
        self.assertEqual(len(onlysynapses), 1)
        self.assertEqual(len(onlynodes), 1)
        c = nest.Create("aeif_psc_alpha", 5)
        dump2 = Dump()
        self.assertNotEqual(thedump, dump2)

    def test_dump_emtpy(self):
        nest.ResetKernel()
        wholedump = Dump()
        self.assertEqual(wholedump, {})
        nodes = Dump(selections=["nodes"])
        self.assertEqual(nodes, {})

    def test_load_repeated(self):
        #checks for growing network size
        nest.ResetKernel()
        a = nest.Create("aeif_psc_alpha", 5)
        b = nest.Create("aeif_psc_alpha", 5)
        nest.Connect(a, b)
        thedump = Dump()
        numload0 = nest.GetKernelStatus("network_size")
        #no reset loadand load twice
        result = Load(thedump)
        numload1 = nest.GetKernelStatus("network_size")
        self.assertGreater(numload1, numload0)
        result = Load(thedump)
        numload2 = nest.GetKernelStatus("network_size")
        self.assertGreater(numload2, numload1)

    def test_load_empty(self):
        nest.ResetKernel()
        numbefore = nest.GetKernelStatus("network_size")
        syns = Load({})
        numafter = nest.GetKernelStatus("network_size")
        self.assertEqual(numbefore, numafter)

    def test_integration(self):
        nest.ResetKernel()
        a = nest.Create("aeif_psc_alpha", 5)
        b = nest.Create("aeif_psc_alpha", 5)
        nest.Connect(a, b)
        thedump = Dump()
        nest.ResetKernel()
        result = Load(thedump)


if __name__ == '__main__':
    unittest.main()
