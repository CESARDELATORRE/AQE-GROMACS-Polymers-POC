from aiida.orm import Int, Float, Bool, Str, List, SinglefileData
from aiida.engine import calcfunction

@calcfunction
def calc_simulation_box_length(molecular_weight_polymer: Float, polymer_count: Int) -> Float:
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

    return Float(box_length)

@calcfunction
def get_polymer_name(monomerfname: Str) -> Str:
    return Str(monomerfname.value.replace('Monomer.pdb', ''))

@calcfunction
def convert_top_to_itp(top: SinglefileData, filename: Str) -> SinglefileData:
    lines = top.get_content().split('\n')
    itp_lines = []
    write = False
    for line in lines:
        if line.startswith('[ moleculetype ]'):
            write = True

        if write:
            if line.startswith('; Include water topology'):
                write = False
                break
            else:
                if line.startswith('Other'):
                    line = line.replace('Other', f'{filename.value[:-4]}')
                itp_lines.append(line)
    return SinglefileData.from_string('\n'.join(itp_lines), filename=filename)

# This will only work for binary blend of polymers
@calcfunction
def get_polymer_count(total_polymer_count: Int, first_polymer_wt_perc: Float, mw_list: List) -> List:
    polymer_count_1 = (first_polymer_wt_perc.value * total_polymer_count.value * mw_list.get_list()[1]) / (100 * mw_list.get_list()[1] + (mw_list.get_list()[1] - mw_list.get_list()[0]) * first_polymer_wt_perc.value)
    polymer_count_2 = total_polymer_count.value - int(polymer_count_1)
    return List([int(polymer_count_1), int(polymer_count_2)])

@calcfunction
def check_insert_molecules(log: SinglefileData, polymer_count: Int) -> Bool:
    lines = log.get_content().split('\n')
    check_lines = [line for line in lines if line.startswith('Added')]
    
    if len(check_lines) > 1:
        raise Exception('ERROR: Multiple lines got selected in the check_lines variable', check_lines)

    wordlist = check_lines[0].split()
    polymer_count_inserted = Int(wordlist[1])
    
    if polymer_count_inserted.value == polymer_count.value:
        return Bool(False)
    else:
        return Bool(True)

@calcfunction
def get_em_mdp() -> SinglefileData:
    return SinglefileData.from_string(
        """
        integrator      = steep
        emtol           = 1000.0
        emstep          = 0.01
        nsteps          = 50000
        nstlist         = 1
        cutoff-scheme   = Verlet
        ns_type         = grid
        coulombtype     = PME
        rcoulomb        = 1.0
        rvdw            = 1.0
        pbc             = xyz
        ld_seed         = 1
        gen_seed        = 1
        """,
        filename='em.mdp',
        )

@calcfunction
def get_npt_mdp(id: Int = None, temperature: Float = None, pressure: Float = None) -> SinglefileData:
    
    id = id if id is not None else Int(0)
    temperature = temperature if temperature is not None else Float(298.15)
    pressure = pressure if pressure is not None else Float(1.0)
    
    mdp_str = f"""
        title                   = NPT Equilibration
        ;define                 = -DPOSRES
        integrator              = md
        dt                      = 0.002
        nsteps                  = 5000
        nstenergy               = 2000
        nstxout-compressed      = 10000
        nstvout                 = 0
        nstlog                  = 1000
        gen_vel                 = yes
        gen_temp                = 298.15
        pbc                     = xyz
        cutoff-scheme           = Verlet
        rlist                   = 1.0
        ns_type                 = grid
        nstlist                 = 10
        coulombtype             = PME
        fourierspacing          = 0.12
        pme_order               = 4
        rcoulomb                = 1.0
        vdwtype                 = Cut-Off
        rvdw                    = 1.0
        DispCorr                = EnerPres
        constraints             = h-bonds
        constraint_algorithm    = lincs
        lincs_iter              = 1
        lincs_order             = 4
        tcoupl                  = v-rescale
        tc-grps                 = System
        ref_t                   = {temperature.value}
        tau_t                   = 0.1
        pcoupl                  = c-rescale
        pcoupltype              = isotropic
        ref_p                   = {pressure.value}
        tau_p                   = 2.0
        ;refcoord-scaling        = com
        compressibility         = 4.5e-5
        """
    
    return SinglefileData.from_string(mdp_str, filename=f'eqnpt-{id.value}.mdp')