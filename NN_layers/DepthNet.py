# DepthNet : focal length 
layer_info = \
{1: {'B': 1, 'K': 64, 'C':3, 'OY': 224, 'OX': 224, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
2: {'B': 1, 'K': 64, 'C':64, 'OY': 224, 'OX': 224, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
3: {'B': 1, 'K': 128, 'C':64, 'OY': 112, 'OX': 112, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
4: {'B': 1, 'K': 128, 'C':128, 'OY': 112, 'OX': 112, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
5: {'B': 1, 'K': 256, 'C':128, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
6: {'B': 1, 'K': 256, 'C':256, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
7: {'B': 1, 'K': 256, 'C':256, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
8: {'B': 1, 'K': 512, 'C':256, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
9: {'B': 1, 'K': 512, 'C':512, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
10: {'B': 1, 'K': 512, 'C':512, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
11: {'B': 1, 'K': 512, 'C':512, 'OY': 14, 'OX': 14, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
12: {'B': 1, 'K': 512, 'C':512, 'OY': 14, 'OX': 14, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
13: {'B': 1, 'K': 512, 'C':512, 'OY': 14, 'OX': 14, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
14: {'B': 1, 'K': 32, 'C':64, 'OY': 224, 'OX': 224, 'FY': 5, 'FX': 5, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 4, 'PX': 4, 'G': 1},
15: {'B': 1, 'K': 64, 'C':32, 'OY': 224, 'OX': 224, 'FY': 5, 'FX': 5, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 4, 'PX': 4, 'G': 1},
16: {'B': 1, 'K': 64, 'C':128, 'OY': 112, 'OX': 112, 'FY': 5, 'FX': 5, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 4, 'PX': 4, 'G': 1},
17: {'B': 1, 'K': 128, 'C':64, 'OY': 112, 'OX': 112, 'FY': 5, 'FX': 5, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 4, 'PX': 4, 'G': 1},
18: {'B': 1, 'K': 128, 'C':256, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
19: {'B': 1, 'K': 256, 'C':128, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
20: {'B': 1, 'K': 256, 'C':512, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
21: {'B': 1, 'K': 512, 'C':256, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
22: {'B': 1, 'K': 256, 'C':512, 'OY': 14, 'OX': 14, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
23: {'B': 1, 'K': 512, 'C':256, 'OY': 14, 'OX': 14, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
24: {'B': 1, 'K': 256, 'C':1024, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
25: {'B': 1, 'K': 128, 'C':512, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 2, 'PX': 2, 'G': 1},
26: {'B': 1, 'K': 64, 'C':256, 'OY': 112, 'OX': 112, 'FY': 5, 'FX': 5, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 4, 'PX': 4, 'G': 1},
27: {'B': 1, 'K': 32, 'C':128, 'OY': 224, 'OX': 224, 'FY': 5, 'FX': 5, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 4, 'PX': 4, 'G': 1},
28: {'B': 1, 'K': 3, 'C':64, 'OY': 224, 'OX': 224, 'FY': 5, 'FX': 5, 'SY': 1, 'SX': 1, 'SFY': 1, 'SFX': 1, 'PY': 4, 'PX': 4, 'G': 1}}
