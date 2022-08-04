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

load "data/1a4k_protein.pdb", protein
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

load "data/1a4k_protein.pdb_points.pdb.gz", points
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
select surf_pocket1, protein and id [10637,10638,10640,10641,10643,10646,10364,10644,10648,11410,11412,11416,11419,11422,10680,10681,10685,10682,10705,10341,10343,10345,10346,10362,11390,10331,10318,10778,10780,8106,10396,8027,8024,11486,10598,10600,8054,8070,10397,11391,11399,11479,11481] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.278,0.353,0.702]
select surf_pocket2, protein and id [3788,3789,3805,3807,4833,4853,4855,4084,4086,4089,4091,4859,4862,4865,4221,4223,3784,3786,4125,4123,4124,3761,4148,4972,4924,4925,1459,1462,1538,1541,4843,4922,4929,4932,4935,4937,4041,3840,3839,4043,1489,1505,4832,4834] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.388,0.361,0.902]
select surf_pocket3, protein and id [3919,657,1371,1628,3982,3977,3957,3958,3993,1365,1367,1402,1405,1408,3991,1591,1594,1597,1598,1584] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.396,0.278,0.702]
select surf_pocket4, protein and id [7222,7936,10476,7932,7967,7970,7971,8156,10514,10515,10534,8149,8155,8159,10503,10505,10471,10548,10474,10475,8193,7217,7220,7221,7213] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.631,0.361,0.902]
select surf_pocket5, protein and id [12414,12416,12378,12380,12383,12358,12390,9184,9185,9186,9210,9213,8731,8730,9231] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.584,0.278,0.702]
select surf_pocket6, protein and id [5631,5639,3940,3943,3937,5111,5113,4714,4712,4716,5083,5630,4738,5607,5616,5625,6034,5122,5139,4675,6036,5945,5949,5998] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.875,0.361,0.902]
select surf_pocket7, protein and id [12008,12012,11944,11947,11896,11900,11903,11916,11922,11905,11911,11914,11993,12004,8404,8379,8392,8400,8410,8421,8415,9813,8460,9801] 
set surface_color,  pcol7, surf_pocket7 
set_color pcol8 = [0.702,0.278,0.627]
select surf_pocket8, protein and id [11679,11696,11670,11718,12164,11271,11273,12506,12593,12551,12555,12188,12191,12173,12182,12591] 
set surface_color,  pcol8, surf_pocket8 
set_color pcol9 = [0.902,0.361,0.682]
select surf_pocket9, protein and id [5823,5827,5833,5855,5859,2164,2165,2166,2152,6129,6131,5821,6133,2645,2621,2666,2639,2620,2619] 
set surface_color,  pcol9, surf_pocket9 
set_color pcol10 = [0.702,0.278,0.439]
select surf_pocket10, protein and id [1850,3221,5337,5339,5343,5346,5386,5387,5390,5447,5451,5436,6176,1814,1827,1831,1837,1841,1845,1856] 
set surface_color,  pcol10, surf_pocket10 
set_color pcol11 = [0.902,0.361,0.439]
select surf_pocket11, protein and id [2823,2382,2827,2458,2459,2422,2940,2955,2957,2959,2885,2937,2491,2457] 
set surface_color,  pcol11, surf_pocket11 
set_color pcol12 = [0.702,0.310,0.278]
select surf_pocket12, protein and id [969,994,997,678,972,982,801,757,751,639,675,1331,1052,1054] 
set surface_color,  pcol12, surf_pocket12 
set_color pcol13 = [0.902,0.522,0.361]
select surf_pocket13, protein and id [9520,9524,9505,9388,9392,9024,9023] 
set surface_color,  pcol13, surf_pocket13 
set_color pcol14 = [0.702,0.502,0.278]
select surf_pocket14, protein and id [6555,5331,6271,6276,6556,6554,6283,6219,6245,6202,6206] 
set surface_color,  pcol14, surf_pocket14 
set_color pcol15 = [0.902,0.765,0.361]
select surf_pocket15, protein and id [7534,7559,7562,7243,7240,7876,7877,7896,7885,7366,7337,7547,7204,7316,7322,7617,7619] 
set surface_color,  pcol15, surf_pocket15 
set_color pcol16 = [0.702,0.690,0.278]
select surf_pocket16, protein and id [10930,11261,10900,10905,10926,10853,10857,10832,10833,11212,10458,10564,10570,10567,10838,11242,10455,10449] 
set surface_color,  pcol16, surf_pocket16 


deselect

orient
