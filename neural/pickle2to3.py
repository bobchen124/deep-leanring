#!/usr/bin/env python3

import pickle

testpkl = pickle.loads(open("sample_weight.pkl", "rb").read())

pickle.dump(testpkl, open("sample_weight2.pkl", "wb"), protocol=2)
