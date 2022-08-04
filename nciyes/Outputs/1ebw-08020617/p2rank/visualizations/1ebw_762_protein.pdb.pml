from pymol import cmd,stored

set depth_cue, 1
set fog_start, 0.4

set_color b_col, [36,36,85]
set_color t_col, [10,10,10]
set bg_rgb_bottom, b_col
set bg_rgb_top, t_col      
set bg_gradient

set  spec_power  =  200
set  spec_refl   =  0

load "data/1ebw_762_protein.pdb", protein
create ligands, protein and organic
select xlig, protein and organic
delete xlig

hide everything, all

color white, elem c
color bluewhite, protein
#show_as cartoon, protein
show surface, protein
#set transparency, 0.15

show sticks, ligands
set stick_color, magenta

load "data/1ebw_762_protein.pdb_points.pdb.gz", points
hide nonbonded, points
show nb_spheres, points
set sphere_scale, 0.2, points
cmd.spectrum("b", "green_red", selection="points", minimum=0, maximum=0.7)


stored.list=[]
cmd.iterate("(resn STP)","stored.list.append(resi)")    # read info about residues STP
lastSTP=stored.list[-1] # get the index of the last residue
hide lines, resn STP

cmd.select("rest", "resn STP and resi 0")

for my_index in range(1,int(lastSTP)+1): cmd.select("pocket"+str(my_index), "resn STP and resi "+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.show("spheres","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_scale","0.4","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_transparency","0.1","pocket"+str(my_index))



set_color pcol1 = [0.361,0.576,0.902]
select surf_pocket1, protein and id [756,769,2330,2332,2861,1924,2830,441,444,450,454,462,465,466,464,436,455,456,751,469,733,457,2814,1681,2829,766,768,2319,2320,2325,767,2815,1984,2314,414,1297,376,1298,1989,1998,2860,2042,1962,1992,2010,2013,2014,1266,1252,1265,493,736,2278,2293,2296,2298,2299,2017,2311,2312,131,132,133,2002,2003,2004,2005] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.329,0.278,0.702]
select surf_pocket2, protein and id [1820,1821,1822,1827,1782,2548,2549,1784,1786,2583,2587,1823,1828,2532,2533,2153,2154,2550,2552] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.698,0.361,0.902]
select surf_pocket3, protein and id [1363,1361,1364,1365,468,1147,1150,1151,1152,938,946,951,1124,1141,1367] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.702,0.278,0.639]
select surf_pocket4, protein and id [2714,2715,2704,2705,2710,2509,2514,2924,2927,2930,2926,2980,2687,2997] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.902,0.361,0.545]
select surf_pocket5, protein and id [272,988,273,274,275,238,239,237,279,280,605,606] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.702,0.353,0.278]
select surf_pocket6, protein and id [468,698,1364,1365,466,1150,904,1151,1152,1153,1367] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.902,0.729,0.361]
select surf_pocket7, protein and id [1731,1554,2622,1585,1586,1587,1735,1746,1751,1900,1714] 
set surface_color,  pcol7, surf_pocket7 


deselect

orient
