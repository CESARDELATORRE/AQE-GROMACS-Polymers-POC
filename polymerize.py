import numpy as np
from aiida.orm import Int, Str, List, Dict, ArrayData, SinglefileData
from aiida.engine import WorkChain, calcfunction

@calcfunction
def get_atom_lines(lines) -> List:
    return List([line for line in lines if line.startswith('ATOM') or line.startswith('HETATM')])

@calcfunction
def get_atom_dict(num: Int, line: Str) -> Dict:
    return Dict({'atom_number': num.value, 'atom_name': line.value[12:16].split()[0], \
                                            'residue_name': line.value[17:20].split()[0], \
                                            'residue_seq_num': int(line.value[22:26].split()[0]), \
                                            'coord': [float(line.value[30:38].split()[0]), \
                                                      float(line.value[38:46].split()[0]), \
                                                      float(line.value[46:54].split()[0])], \
                                            'element': line.value[76:78].split()[0]})

@calcfunction
def copy_atomdict(curratom_dict: Dict) -> Dict:
    newatom_dict = Dict()
    newatom_dict.set_dict(curratom_dict)
    return newatom_dict

@calcfunction
def update_atom_number(curratom_dict: Dict, atom_number: Int) -> Dict:
    return Dict({'atom_number': atom_number.value, \
                         'atom_name': curratom_dict['atom_name'], \
                         'residue_name': curratom_dict['residue_name'], \
                         'residue_seq_num': curratom_dict['residue_seq_num'], \
                         'coord': curratom_dict['coord'], \
                         'element': curratom_dict['element']})

@calcfunction
def update_residue_seq_num(curratom_dict: Dict, residue_seq_num: Int) -> Dict:
    return Dict({'atom_number': curratom_dict['atom_number'], \
                 'atom_name': curratom_dict['atom_name'], \
                 'residue_name': curratom_dict['residue_name'], \
                 'residue_seq_num': residue_seq_num.value, \
                 'coord': curratom_dict['coord'], \
                 'element': curratom_dict['element']})

@calcfunction
def update_residue_name(curratom_dict: Dict, residue_name: Str) -> Dict:
    return Dict({'atom_number': curratom_dict['atom_number'], \
                 'atom_name': curratom_dict['atom_name'], \
                 'residue_name': residue_name.value, \
                 'residue_seq_num': curratom_dict['residue_seq_num'], \
                 'coord': curratom_dict['coord'], \
                 'element': curratom_dict['element']})

@calcfunction
def update_coord(curratom_dict: Dict, coord: List) -> Dict:
    return Dict({'atom_number': curratom_dict['atom_number'], \
                 'atom_name': curratom_dict['atom_name'], \
                 'residue_name': curratom_dict['residue_name'], \
                 'residue_seq_num': curratom_dict['residue_seq_num'], \
                 'coord': coord.get_list(), \
                 'element': curratom_dict['element']})

@calcfunction
def get_pdbstr(curratom_dict: Dict) -> Str:
    pdb_str = "ATOM  %5d %-4s %3s  %4d    %8.3f%8.3f%8.3f                      %2s" \
    % (curratom_dict['atom_number']+1, curratom_dict['atom_name'], 
       curratom_dict['residue_name'], curratom_dict['residue_seq_num']+1, \
       curratom_dict['coord'][0], curratom_dict['coord'][1], curratom_dict['coord'][2], \
       curratom_dict['element'])
    return Str(pdb_str)

@calcfunction
def get_unit_vector(p1: ArrayData, p2: ArrayData) -> ArrayData:

    vec = p1.get_array() - p2.get_array()
    vec /= np.linalg.norm(vec)

    return ArrayData(vec)

@calcfunction
def get_rotation_matrix(vec1: ArrayData, vec2: ArrayData) -> ArrayData:

    # Calculate the angle between vec1 and vec2 and rotation angle to align vec2 to vec1
    dot_product = np.dot(vec1.get_array(), vec2.get_array())
    rotation_angle = np.arccos(dot_product)
    print('rangle = ', rotation_angle)

    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    
    # Calculate the rotation axis (cross product)
    rotation_axis = np.cross(vec1.get_array(), vec2.get_array())
    rotation_axis /= np.linalg.norm(rotation_axis)

    rotation_matrix = np.array([
        [cos_angle + rotation_axis[0] * rotation_axis[0] * (1 - cos_angle), \
        rotation_axis[0] * rotation_axis[1] * (1 - cos_angle) - rotation_axis[2] * sin_angle, \
        rotation_axis[0] * rotation_axis[2] * (1 - cos_angle) + rotation_axis[1] * sin_angle],
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_angle) + rotation_axis[2] * sin_angle, \
        cos_angle + rotation_axis[1] * rotation_axis[1] * (1 - cos_angle), \
        rotation_axis[1] * rotation_axis[2] * (1 - cos_angle) - rotation_axis[0] * sin_angle],
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_angle) - rotation_axis[1] * sin_angle, \
        rotation_axis[2] * rotation_axis[1] * (1 - cos_angle) + rotation_axis[0] * sin_angle, 
        cos_angle + rotation_axis[2] * rotation_axis[2] * (1 - cos_angle)]
    ])
    return ArrayData(rotation_matrix)

@calcfunction
def rotate_coord(atom_coord: ArrayData, ref_coord: ArrayData, rotation_matrix: ArrayData) -> ArrayData:

    ref_atom_vec = atom_coord.get_array() - ref_coord.get_array()
    
    rotated_coord_vec = np.dot(rotation_matrix.get_array(), ref_atom_vec)
    new_atom_coord = ref_coord.get_array() + rotated_coord_vec

    return ArrayData(new_atom_coord)
    
class polymerize(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('monomer', valid_type = SinglefileData)
        spec.input('monomer_count', valid_type = Int)
        spec.output('polymer', valid_type = SinglefileData)
        spec.outline(cls.make_polymer, cls.result)

    def make_polymer(self):
        print('c1')
        monomer_atom_lines = get_atom_lines(self.inputs.monomer.get_content().split('\n'))
        print('c2')
        monomer_all_atom_list = List()
        for num, line in enumerate(monomer_atom_lines.get_list()):
            monomer_all_atom_list.append(get_atom_dict(Int(num), line))
        print('c3')
        print(monomer_all_atom_list.get_list())
        print('c4')

        polymer_all_atom_list = List()
        polymer_remove_atom_index_list = List()
        print('c5')

        # Add monomers to the polymer chain
        polymer_atom_count = 0
        for imonomer in range(self.inputs.monomer_count.value):
            print('Start ', imonomer)
            monomer_atom_count = len(monomer_all_atom_list)
            # first monomer
            if imonomer == 0:
                for iatom in monomer_all_atom_list:
                    curratom = copy_atomdict(iatom)
                    curratom = update_atom_number(curratom, Int(polymer_atom_count))
                    curratom = update_residue_seq_num(curratom, Int(imonomer))
                    
                    polymer_all_atom_list.append(curratom)
                    
                    if iatom['atom_name'] == 'HW3':
                        polymer_remove_atom_index_list.append(polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_number'])
                    polymer_atom_count += 1
            else:
                # get the CW of last monomer
                print('p1')
                cw_index = -1
                for iatom in polymer_all_atom_list:
                    if iatom['atom_name'] == 'CW':
                        cw_index = iatom['atom_number']

                if cw_index < 0:
                    raise ValueError('CW atom is not found.')
        
                # get the HA3 of model monomer
                ha3_index = -1
                for iatom in monomer_all_atom_list:
                    if iatom['atom_name'] == 'HA3':
                        ha3_index = iatom['atom_number']
                
                if ha3_index < 0:
                    raise ValueError('HA3 atom is not found.')

                dtranslate = np.array(polymer_all_atom_list[cw_index]['coord']) \
                - np.array(monomer_all_atom_list[ha3_index]['coord'])

                print('p2')
                # put the next monomer in the polymer + translation
                for iatom in monomer_all_atom_list:
                    curratom = copy_atomdict(iatom)
                    curratom = update_atom_number(curratom, Int(polymer_atom_count))
                    curratom = update_residue_seq_num(curratom, Int(imonomer))

                    coord = np.array(iatom['coord']) + dtranslate
                    curratom = update_coord(curratom, List(coord.tolist()))

                    if imonomer != self.inputs.monomer_count.value - 1:
                        curratom = update_residue_name(curratom, Str(iatom['residue_name'][:-1] + '2'))
                    else:
                        curratom = update_residue_name(curratom, Str(iatom['residue_name'][:-1] + '3'))
            
                    polymer_all_atom_list.append(curratom)
                    
                    if polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_name'] == 'HA3':
                        polymer_remove_atom_index_list.append(polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_number'])
                    elif polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_name'] == 'HW3':
                        if polymer_all_atom_list[len(polymer_all_atom_list)-1]['residue_seq_num'] != self.inputs.monomer_count.value - 1:
                            polymer_remove_atom_index_list.append(polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_number'])
                    
                    polymer_atom_count += 1

                print('p3')
                # rotation starts here
                # CW.coord == HA3.coord
                # get the CA of last monomer
                ca_index = -1
                for iatom in polymer_all_atom_list:
                    if iatom['atom_name'] == 'CA':
                        ca_index = iatom['atom_number']
                        
                if ca_index < 0:
                    raise ValueError('CA atom is not found.')

                # get the HW3 of previous monomer
                hw3_index = -1
                for iatom in polymer_all_atom_list:
                    if iatom['atom_name'] == 'HW3' and iatom['residue_seq_num'] == imonomer - 1:
                        hw3_index = iatom['atom_number']
                        
                if hw3_index < 0:
                    raise ValueError('HW3 atom is not found.')

                cw_hw3_unit_vec = \
                get_unit_vector(ArrayData(np.array(polymer_all_atom_list[hw3_index]['coord'])), \
                                ArrayData(np.array(polymer_all_atom_list[cw_index]['coord'])))
        
                cw_ca_unit_vec = \
                get_unit_vector(ArrayData(np.array(polymer_all_atom_list[ca_index]['coord'])), \
                                ArrayData(np.array(polymer_all_atom_list[cw_index]['coord'])))

                print('p4')
                rotation_matrix = get_rotation_matrix(cw_hw3_unit_vec, cw_ca_unit_vec)

                print('p5')
                for index, iatom in enumerate(polymer_all_atom_list):
                    if iatom['residue_seq_num'] == imonomer and iatom['atom_name'] != '':
                        coord = rotate_coord(ArrayData(np.array(iatom['coord'])), \
                                             ArrayData(np.array(polymer_all_atom_list[ca_index]['coord'])), \
                                             rotation_matrix).get_array().tolist()
                        polymer_all_atom_list[index] = update_coord(polymer_all_atom_list[index], coord)
            print('Done ', imonomer)
        polymer_all_atom_lines = List()
        for iatom in polymer_all_atom_list:
            if iatom['atom_number'] not in polymer_remove_atom_index_list.get_list():
                polymer_all_atom_lines.append(get_pdbstr(iatom).value)
                print(get_pdbstr(iatom).value)
        self.ctx.polymer = SinglefileData.from_string('\n'.join(polymer_all_atom_lines), filename='polymer.pdb')

    def result(self):
        self.out('polymer', self.ctx.polymer)