import os
import numpy as np
import pandas as pd

from aiida.orm import Int, Float, Str, List, Dict, ArrayData, SinglefileData
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
def get_unit_vector(pos1: ArrayData, pos2: ArrayData) -> ArrayData:

    vec = pos1.get_array('vec') - pos2.get_array('vec')
    vec /= np.linalg.norm(vec)

    res = ArrayData()
    res.set_array('vec', vec)

    return res

@calcfunction
def get_rotation_matrix(vec1: ArrayData, vec2: ArrayData) -> ArrayData:

    # Calculate the angle between vec1 and vec2 and rotation angle to align vec2 to vec1
    dot_product = np.dot(vec1.get_array('vec'), vec2.get_array('vec'))
    rotation_angle = np.arccos(dot_product)
    #print('rotation angle = ', rotation_angle)

    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    
    # Calculate the rotation axis (cross product)
    rotation_axis = np.cross(vec2.get_array('vec'), vec1.get_array('vec'))
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

    res = ArrayData()
    res.set_array('vec', rotation_matrix)
    return res

@calcfunction
def rotate_coord(atom_coord: ArrayData, ref_coord: ArrayData, rotation_matrix: ArrayData) -> ArrayData:

    ref_atom_vec = atom_coord.get_array('vec') - ref_coord.get_array('vec')
    
    rotated_coord_vec = np.dot(rotation_matrix.get_array('vec'), ref_atom_vec)
    new_atom_coord = ref_coord.get_array('vec') + rotated_coord_vec

    res = ArrayData()
    res.set_array('vec', new_atom_coord)

    return res
    
class PolymerizeWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('monomer', valid_type = SinglefileData)
        spec.input('polymer_connection_point_list', valid_type = List)
        spec.input('monomer_count', valid_type = Int)
        spec.output('polymer', valid_type = SinglefileData)
        spec.output('polymer_molecular_weight', valid_type = Float)
        spec.outline(cls.make_polymer, cls.result)

    def make_polymer(self):
        monomer_atom_lines = get_atom_lines(self.inputs.monomer.get_content().split('\n'))
        monomer_all_atom_list = List()
        for num, line in enumerate(monomer_atom_lines.get_list()):
            monomer_all_atom_list.append(get_atom_dict(Int(num), line))
        
        polymer_all_atom_list = List()
        polymer_remove_atom_index_list = List()

        #['CW', 'HW3', 'HA3', 'CA']
        cw_atom_name = self.inputs.polymer_connection_point_list.get_list()[0]
        hw3_atom_name = self.inputs.polymer_connection_point_list.get_list()[1]
        ha3_atom_name = self.inputs.polymer_connection_point_list.get_list()[2]
        ca_atom_name = self.inputs.polymer_connection_point_list.get_list()[3]
        #print(cw_atom_name, hw3_atom_name, ha3_atom_name, ca_atom_name)

        # get the HA3 of model monomer
        ha3_index = -1
        for iatom in monomer_all_atom_list:
            if iatom['atom_name'] == ha3_atom_name:
                ha3_index = iatom['atom_number']
                break
        
        if ha3_index < 0:
            raise ValueError('HA3 atom is not found.')

        # get the CA of model monomer
        ca_index = -1
        for iatom in monomer_all_atom_list:
            if iatom['atom_name'] == ca_atom_name:
                ca_index = iatom['atom_number']
                break

        if ca_index < 0:
            raise ValueError('CA atom is not found.')

        # get coordinate of connection point (lies on CA-HA3 vector with a CA-Connection point distance of 1.58)
        pos1 = ArrayData()
        pos1.set_array('vec', np.array(monomer_all_atom_list[ha3_index]['coord']))
        pos2 = ArrayData()
        pos2.set_array('vec', np.array(monomer_all_atom_list[ca_index]['coord']))
        ca_ha3_unit_vec = get_unit_vector(pos1, pos2)
        ha3_coord = np.array(monomer_all_atom_list[ca_index]['coord']) + ca_ha3_unit_vec.get_array('vec') * 1.58

        #print(monomer_all_atom_list[ha3_index]['coord'], monomer_all_atom_list[ca_index]['coord'], ha3_coord)
        
        # Add monomers to the polymer chain
        polymer_atom_count = 0
        print(f'Polymerization starts for {self.inputs.monomer.filename} ->')
        for imonomer in range(self.inputs.monomer_count.value):
            print(imonomer+1, end='\r')
            #print('Start ', imonomer)
            monomer_atom_count = len(monomer_all_atom_list)
            # first monomer
            if imonomer == 0:
                for iatom in monomer_all_atom_list:
                    curratom = copy_atomdict(iatom)
                    curratom = update_atom_number(curratom, Int(polymer_atom_count))
                    curratom = update_residue_seq_num(curratom, Int(imonomer))
                    
                    polymer_all_atom_list.append(curratom)
                    
                    if iatom['atom_name'] == hw3_atom_name:
                        polymer_remove_atom_index_list.append(polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_number'])
                    polymer_atom_count += 1
            else:
                # get the CW of last monomer
                cw_index = -1
                for iatom in polymer_all_atom_list:
                    if iatom['atom_name'] == cw_atom_name:
                        cw_index = iatom['atom_number']

                if cw_index < 0:
                    raise ValueError('CW atom is not found.')

                dtranslate = np.array(polymer_all_atom_list[cw_index]['coord']) - ha3_coord

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

                    if polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_name'] == ha3_atom_name:
                        polymer_remove_atom_index_list.append(polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_number'])
                    elif polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_name'] == hw3_atom_name:
                        if polymer_all_atom_list[len(polymer_all_atom_list)-1]['residue_seq_num'] != self.inputs.monomer_count.value - 1:
                            polymer_remove_atom_index_list.append(polymer_all_atom_list[len(polymer_all_atom_list)-1]['atom_number'])
                    
                    polymer_atom_count += 1

                #for iatom in polymer_all_atom_list:
                #    print(get_pdbstr(iatom).value)
                #print('before rotate')

                # rotation starts here
                # CW.coord == HA3.coord
                # get the CA of last monomer and HW3 of previous monomer
                ca_index = -1
                hw3_index = -1
                for iatom in polymer_all_atom_list:
                    if iatom['atom_name'] == ca_atom_name:
                        ca_index = iatom['atom_number']
                    if iatom['atom_name'] == hw3_atom_name and iatom['residue_seq_num'] == imonomer - 1:
                        hw3_index = iatom['atom_number']
                        
                if ca_index < 0:
                    raise ValueError('CA atom is not found.')

                pos1 = ArrayData()
                pos1.set_array('vec', np.array(polymer_all_atom_list[hw3_index]['coord']))
                pos2 = ArrayData()
                pos2.set_array('vec', np.array(polymer_all_atom_list[cw_index]['coord']))
                cw_hw3_unit_vec = get_unit_vector(pos1, pos2)

                pos1 = ArrayData()
                pos1.set_array('vec', np.array(polymer_all_atom_list[ca_index]['coord']))
                cw_ca_unit_vec = get_unit_vector(pos1, pos2)
                
                rotation_matrix = get_rotation_matrix(cw_hw3_unit_vec, cw_ca_unit_vec)

                for index, iatom in enumerate(polymer_all_atom_list):
                    #print(iatom['atom_number'])
                    if iatom['residue_seq_num'] == imonomer and iatom['atom_name'] != '':
                        pos1 = ArrayData()
                        pos1.set_array('vec', np.array(iatom['coord']))
                        coord = rotate_coord(pos1, pos2, rotation_matrix).get_array('vec').tolist()
                        polymer_all_atom_list[index] = update_coord(polymer_all_atom_list[index], coord)
            #print('Done ', imonomer)
        polymer_all_atom_lines = List()
        
        self.ctx.polymer_molecular_weight = Float(0.0)
        dataframe_elements = pd.read_csv(os.getcwd() + '/elements.csv', index_col = None)

        print('')
        #print('remove = ', polymer_remove_atom_index_list.get_list())
        for iatom in polymer_all_atom_list:
            if iatom['atom_number'] not in polymer_remove_atom_index_list.get_list():
                self.ctx.polymer_molecular_weight += \
                Float(dataframe_elements.loc[dataframe_elements['Symbol'] == \
                      iatom['element'], 'AtomicMass'].iloc[0])
                polymer_all_atom_lines.append(get_pdbstr(iatom).value)
                print(get_pdbstr(iatom).value)
        self.ctx.polymer = SinglefileData.from_string('\n'.join(polymer_all_atom_lines), filename='polymer.pdb')

    def result(self):
        self.out('polymer', self.ctx.polymer)
        self.out('polymer_molecular_weight', self.ctx.polymer_molecular_weight)