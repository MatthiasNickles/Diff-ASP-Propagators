% Solving MSE cost functions with Clingo + Python using propagators
% Version 0.1

% Copyright (c) 2018 Matthias Nickles
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%   http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

#script(python)

import clingo
import re
import random
import sys
import ad  # ad-1.3.2 (for automatic differentiation, https://pypi.org/project/ad/#description) 
from ad import adnumber
import copy

branch_param_lit = sys.maxint
param_atoms = {}
__inner_costs = {}
h = 0.0001
# for psi and max_trials see below
    
class Propagator:

    solver_lits = {}

    def init(self, init):
        global param_atoms
        
        # Initializing propagator...
                
        for atomproglit, atom in param_atoms.iteritems():
            solver_lit = init.solver_literal(atomproglit)    
            Propagator.solver_lits[atomproglit] = solver_lit
            init.add_watch(solver_lit)
            init.add_watch(-solver_lit)
            
        init.check_mode = clingo.PropagatorCheckMode.Fixpoint     
    
    def propagate(self, control, changes):
        global branch_param_lit 
        global param_atoms

        if branch_param_lit != sys.maxint:
            if branch_param_lit > 0:
                control.add_clause([Propagator.solver_lits[branch_param_lit]], True)        
            else:    
                control.add_clause([-Propagator.solver_lits[abs(branch_param_lit)]], True)     
    
def main(prg):
    global current_cost
    global branch_param_lit     
    global param_atoms
    global __inner_costs
 
    psi = 1.0E-4  # cost function convergence threshold (accuracy)   

    max_trials =  1000
    
    prg.configuration.solver.sign_def = "rnd"
    #prg.configuration.solver.rand_freq = 0.7        
    prg.configuration.solver.seed = random.randint(0, sys.maxint)
    
    models = []
           
    def __cost(freqs):
        return __cost_ad(freqs, "")
        
    def __cost_ad(freqs, atom_x):  # using automatic differentiation
    
        ps = [ (adnumber(freqs['a']), 'a'), (adnumber(freqs['b']), 'b') ] 
        
        c = ( (1*(0.2-ps[0][0])**2) + (1*(0.6-ps[1][0])**2)) / 2    # example cost function (0.2 = weight of a, 0.6 = weight of b)
        
        if atom_x == "":
            return c
        else:    
            pxi = next(i for i,v in enumerate(ps) if v[1] == atom_x)
            return c.d(ps[pxi][0])
   
    def __cost_upd(freqs, upd_atom):  # using single point numerical differentiation
        global h
        
        if len(freqs) == 0:
            return sys.float_info.max
        else:  
            freqs_upd = freqs.copy() 
            freqs_upd[upd_atom] += h
            return __cost(freqs_upd)    
            
    def update_cost():    
        global branch_param_lit
        global param_atoms
        global h
      
        use_auto_diff = False  # choose numerical or automatic differentiation (using the ad package)
        
        freqs = {}
      
        for atomlit, atom in param_atoms.iteritems():
            if len(models) == 0:
                freqs[atom] = 0
            else:
                freqs[atom] = float(sum(ms.find(" " + atom + " ") != -1 for ms in models)) / float(len(models))  
                # linear-time freqs updates possible too (straightforward)
                   
        current_cost = __cost(freqs)
        print("Current cost = " + str(current_cost))

        min_diff_lit = (param_atoms.keys()[0], sys.float_info.max)
        
        for atomlit, atom in param_atoms.iteritems():  # we search for the minimum partial derivative

            diff_p = __cost_ad(freqs, atom) if use_auto_diff else (__cost_upd(freqs, atom) - current_cost) / h
            
            if diff_p < min_diff_lit[1]:
                min_diff_lit = (atomlit, diff_p) 
            diff_n = -diff_p
            if diff_n < min_diff_lit[1]:
                min_diff_lit = (-atomlit, diff_n)                   
        
        branch_param_lit = min_diff_lit[0]
                
        return current_cost        
        
    def on_model(m):    
        posAtoms = m.symbols(atoms=True)
        ms = " ".join(map(str, m.symbols(atoms=True)))
        models.append(" " + ms + " ")  # we add the set of atoms in m as a new model to the sample multiset                 
        
    prg.configuration.solve.models = 1

    prg.register_propagator(Propagator())  # see class Propagator

    prg.ground([("base", [])])
        
    for atom in prg.symbolic_atoms:
        atom_str = str(atom.symbol)
        if atom_str in [ "b", "a" ]:
            param_atoms[atom.literal] = atom_str
    
    i = 1
    while i <= max_trials:          
        
        print("Model sampling iteration: " + str(i) + "...")
        
        ret = prg.solve(on_model=on_model)
        print(ret)
        print("#models sampled so far: " + str(len(models)))
                
        current_cost = update_cost()
        
        if current_cost <= psi:  # target accuracy reached
            i = sys.maxint
        else:
            i += 1
                        
    if i != sys.maxint:
        print("Warning: No convergence with psi = " + str(psi) + " and max_trials = " + str(max_trials))

    # return 'models', or compute further models st. current_cost<=psi, or sample uniformly from 'models'    
#end.
