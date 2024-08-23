# -*- coding: utf-8 -*-
'''
@File   :  vis.py
@Time   :  2024/08/17 22:02
@Author :  Yufan Liu
@Desc   :  visualize the result if structure is possible
'''

from pymol import cmd
show_as_doable = ['cartoon', 'surface', 'sticks', 'mesh']
color_doable = ['red', 'blue','green', 'magenta']

def all_integers(lst):
    return all(isinstance(i, int) for i in lst)

def visualize(pdb_file, 
              chain:str, 
              result:list, 
              out_file,
              show_as='surface', 
              color='red', 
              select_only=False,
              ligand_id:str=None
              ):
    """
    assume only 1 chain for protein/ligand, will be change later.
    """
    assert show_as in show_as_doable, f"Viable show modes: {show_as_doable}"
    assert color in color_doable, f"Viable colors: {color_doable}"
    
    cmd.load(pdb_file.lower(), 'pdb_file')
    cmd.show(show_as, f'chain {chain}')
    cmd.color('gray90', f'chain {chain}')
    
    residue_list = result
    all_chains = cmd.get_chains('pdb_file')
    
    if all_integers(residue_list):
        for resi in residue_list:
            selection = f'chain {chain} and resi {resi}'
            cmd.color(color, selection)
    else:
        # color by b-facotr
        for i, resi in enumerate(range(1, len(residue_list) + 1)):
            cmd.alter(f"resi {resi}", f"b={residue_list[i]}")
            cmd.spectrum("b", "blue_red", selection=f"chain {chain}")

    if select_only and ligand_id:
        raise RuntimeError(f"set select_only to False if you want to save ligand")
    
    if select_only:
        chain = chain.split(",")
        remove = [x for x in all_chains if x not in chain]
        selection = cmd.remove(f'chain {"+".join(remove)} and pdb_file')
        cmd.save(out_file)
        cmd.quit()
        return
    elif ligand_id:
        try:
            ligand_id = ligand_id.split(",")
            ligand_id.append(chain)
            remove = [x for x in all_chains if x not in ligand_id]
            selection = cmd.remove(f'chain {"+".join(remove)} and pdb_file')
            cmd.save(out_file)
            cmd.quit()
        except Exception as e:
            print(e, "Make sure input id looks like: A,B,C, split with comma")
        return
    else:
        cmd.save(out_file)
        cmd.quit()
        return 
    
    