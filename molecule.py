import os
import numpy as np
import pandas as pd

from aiida.orm import Float, SinglefileData # Int, Str, List, Dict, ArrayData, SinglefileData
from aiida.engine import WorkChain, calcfunction

@calcfunction
def calc_molecular_weight_polymer(polymer: SinglefileData) -> Float:
    elecsvfile = os.getcwd() + '/elements.csv'
    dataframe_elements = pd.read_csv(elecsvfile, index_col = None)
    
    lines = polymer.get_content().split('\n')
    atom_lines = [line for line in lines if line.startswith('ATOM') or line.startswith('HETATM')]

    # Calculate molecular weight
    molecular_weight_polymer = 0.0
    for line in atom_lines:
        # Find element and atomic mass from the periodic table
        elestr = line[12:14]

        if elestr[0] == ' ':
            if elestr[1].isalpha() and elestr[1].isupper():
                ele = elestr[1]
            else:
                ele = ' '
        elif elestr[0].isdigit():
            if elestr[1].isalpha() and elestr[1].isupper():
                ele = elestr[1]
            else:
                ele = ' '
        elif elestr[0].isalpha() and elestr[0].isupper():
            if elestr[1].isdigit():
                ele = elestr[0]
            else:
                ele = elestr
        else:
            ele = ' '

        try:
            molecular_weight_polymer += float(dataframe_elements.loc[dataframe_elements['Symbol'] == ele, 'AtomicMass'].iloc[0])
        except:
            raise Exception(f'ERROR: Atom type of {elestr} not found.')

    return orm.Float(molecular_weight_polymer)

@engine.calcfunction
def calc_simulation_box_length(molecular_weight_polymer: orm.Float, polymer_count: orm.Int) -> orm.Float:
    molecular_weight = molecular_weight_polymer.value * polymer_count.value
    
    # Default density (0.4 g/cm3)
    density = 0.4
    avogadro_number = 6.022e23  # molecules/mol
    cm_to_nm = 1e7  # cm to nm conversion factor
    
    # Step 1: Mass of one molecule
    mass_per_molecule = molecular_weight / avogadro_number  # g
    
    # Step 2: Volume of cubic box
    volume = mass_per_molecule / density  # cm³
    
    # Step 3: Side length of the cubic box
    box_length = volume**(1/3)  # cm
    box_length = box_length * cm_to_nm  # nm

    return orm.Float(box_length)