% use with clingo-python propdiff_1.py.lp propdiff_bgk_1.lp

0{a}1. % spanning rule for parameter atom a 
0{b}1.
:- a, b. % an example for a hard rule
