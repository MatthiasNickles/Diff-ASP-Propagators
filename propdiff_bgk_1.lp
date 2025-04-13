% Use with clingo-python propdiff_1.py.lp propdiff_bgk_1.lp

% For setting atom weights (probabilities) and loss function, see propdiff_1.py.lp 

0{a}1. % spanning rule for parameter atom a 
0{b}1.
:- a, b. % an example for a hard rule
