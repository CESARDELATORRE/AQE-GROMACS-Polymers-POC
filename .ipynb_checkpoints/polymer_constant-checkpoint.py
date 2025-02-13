# aiida packages
from aiida.orm import Int, Str, Dict, List
from aiida.engine import calcfunction

@calcfunction
def get_pdb(key: Int) -> Str:
    pdb_dict = Dict(dict={
        '1': 'tPBMonomer.pdb',
        '2': 'PVAMonomer.pdb'
    }).store()

    result = Str(pdb_dict.get_dict().get(f'{key.value}', "Error: Polymer is not in the current database."))
    
    return result

# Polymerization point
'''
4 elements in the list is given as follows:
    - element 1 and element 2 represent the last two atoms bonded where element 2 will be replace by next monomer element.
    - element 3 and element 4 represent the first two atoms bonded where element 3 will be replace by previous monomer element.
'''
@calcfunction
def get_connection_point(key: Int) -> List:
    connection_dict = Dict(dict={
        '1': ['CW', 'HW3', 'HA3', 'CA'],
        '2': ['CB', 'HB2', 'HA3', 'CA']
    }).store()

    result = List(connection_dict.get_dict().get(f'{key.value}', "Error: Polymer is not in the current database."))

    return result

@calcfunction
def get_classified_property_list(prop_list: List) -> List:
    cls_prop_list = List([])
    cls_prop_list.append([prop for prop in prop_list if prop in ['Potential', 'Density']])
    cls_prop_list.append([prop for prop in prop_list if prop in ['Tg']])

    return cls_prop_list

@calcfunction
def get_all_gromacs_property_list(secondary_prop_list: List) -> List:
    prop_dict = Dict(dict={
        'Tg': ['Density'],
    }).store()

    prop_list = List([])
    for key in secondary_prop_list.get_list():
        temp_prop_list = prop_dict.get_dict().get(key, "Error: Secondary property is not available in the .")
        for prop in temp_prop_list:
            if prop not in prop_list:
                prop_list.append(prop)
    return prop_list
    