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

load "data/1g35_776_protein.pdb", protein
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

load "data/1g35_776_protein.pdb_points.pdb.gz", points
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
select surf_pocket1, protein and id [376,1998,1999,2004,2335,413,414,1976,1313,2876,1977,2007,2028,435,440,444,1312,771,777,2340,2341,2344,2346,436,441,450,465,2830,1695,1939,2844,2845,454,462,466,455,456,766,763,457,748,750,469,2829,1696,1280,1281,1267,492,493,2347,751,1198,2329,2311,2313,2314,2326,2327,2328,2382,2383,784,2334,2336,2057,782,2293,2013,133,2017,2020,2025,2032] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.329,0.278,0.702]
select surf_pocket2, protein and id [2031,2729,2730,2731,2518,2719,2524,2725,2016,2939,2942,2945,2941,3012,2702,3009,2529] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.698,0.361,0.902]
select surf_pocket3, protein and id [1821,2151,1835,1836,1837,1838,1842,2563,2547,2169,2548,2168,2136,1799,1800,1801,1802,1843,1797,2560,2542,2564,2567] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.702,0.278,0.639]
select surf_pocket4, protein and id [1376,1379,468,1382,1378,1449,1166,1162,1167,968,1139,1156] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.902,0.361,0.545]
select surf_pocket5, protein and id [984,997,1000,1001,1002,1004,588,605,606,272,273,274,275,279,573] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.702,0.353,0.278]
select surf_pocket6, protein and id [796,797,798,803,2399,799,804,2800,2801,2802,816,818,773,2830,2831] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.902,0.729,0.361]
select surf_pocket7, protein and id [166,183,187,198,1074,202,203,37,39,6,38] 
set surface_color,  pcol7, surf_pocket7 


deselect

orient
