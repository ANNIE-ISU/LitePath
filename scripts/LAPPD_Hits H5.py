"""Template: load batch output, group hits by event and detector.

Usage:
    python scripts/load_batch.py [--hits results/photon_hits.parquet]
                                 [--pmts results/pmt_responses.parquet]

The output Parquet files have these columns:

    photon_hits.parquet:
        event_id, detector_system, detector_index,
        local_u, local_v, arrival_time, wavelength

    pmt_responses.parquet (only with --pmt-response):
        event_id, pmt_index, charge, time, n_hits

    muon_truth.parquet:
        event_id, pos_x/y/z, t0, dir_x/y/z, theta_deg, phi_deg,
        track_length_mm, n_generated, n_detected

Detector system codes:
    0 = PMT
    1 = LAPPD (default rectangular)
    2 = LAPPD (ANNIE housing)

Detector index maps to the geometry's detector registry (stable IDs).

Muon direction conventions:
    theta_deg: polar angle from vertical (0 = upward, 180 = downward)
    phi_deg:   azimuthal angle in XY plane (arctan2(y, x))
"""

#NOTE: IF VIEWING ON GITHUB THE "output.h5" FILE IS LIKELY MISSING. THIS CODE NEEDS THAT FILE TO RUN! The file was left out due to its large size being incompatible with github file size limits

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

#Below is imports for plotting
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib import patches
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

from annieray.io_h5 import load_table


def main() -> None:
    if len(sys.argv) < 2:
        h5_path = Path("results")
        h5_path2 = Path("results") #This sets up the grab for the second data file
    else:
        h5_path = Path(sys.argv[1])
        h5_path2 = Path(sys.argv[1]) #Not sure if this is correct but its been working

    if h5_path.suffix == ".h5":
        pass
    else:
        h5_path = h5_path / "output.h5"
        #h5_path2 = h5_path2 / "output_no_struct.h5" #This is the second data file used for the ratios

    # ── Load ──────────────────────────────────────────────────────
    hits = load_table(h5_path, "photon_hits")
    if hits.empty:
        print("No photon_hits found.")
        return

    muons = load_table(h5_path, "muon_truth")
    if muons.empty:
        print("No muon_truth found.")
        return

    """
   # ── Loading second data set ──────────────────────────────────────────────────────
    
    #Need to check how to make this the proper way, kind of making a guess that seems to be working
    
    hits2 = load_table(h5_path2, "photon_hits")
    if hits2.empty:
        print("No photon_hits found.")
        return

    muons2 = load_table(h5_path2, "muon_truth")
    if muons2.empty:
        print("No muon_truth found.")
        return
    

    #── END of second data set loading ──────────────────────────────────────────────────────
    """

    ratioCondition = input("Plot the ratio of structure vs no structure? (y/n)")
    print(ratioCondition)
    
    #Creating LAPPD hit data
    LAPPD_Indices = [132, 133, 134]
    lappd_hits = hits[hits["detector_index"].isin(LAPPD_Indices)]
    counts = lappd_hits.groupby(["event_id", "detector_index"]).size().reset_index(name="n_hits")
    pivoted = counts.pivot(index="event_id", columns="detector_index", values="n_hits").fillna(0).astype(int)
    pivoted.columns = [f"n_pmt{c}" for c in pivoted.columns]

    tablePhotonsPerMuonHits = muons[["event_id", "pos_x", "pos_z","dir_x","dir_y","dir_z"]].merge(pivoted, on="event_id", how="left").fillna(0)
    
    #Unnormalizing the direction columns
    tablePhotonsPerMuonHits["dir_x"] = (tablePhotonsPerMuonHits["dir_x"] / tablePhotonsPerMuonHits["dir_y"]).round(3) 
    tablePhotonsPerMuonHits["dir_z"] = (tablePhotonsPerMuonHits["dir_z"] / tablePhotonsPerMuonHits["dir_y"]).round(3)
    tablePhotonsPerMuonHits["dir_y"] = (tablePhotonsPerMuonHits["dir_y"] / tablePhotonsPerMuonHits["dir_y"]).round(3) #This has to be last since its value ends up being 1

    
    #Making the table into a numpy array
    tablePhotonsPerMuonHits.to_numpy()

    #Checking the sorted data by making it into a text file
    tablePhotonsPerMuonHits.to_csv('tablePhotonPerMuonHits.txt', sep=' ', index=False)


    #Getting the needed data for the event with structure
    data = tablePhotonsPerMuonHits[["n_pmt132", "n_pmt133", "n_pmt134"]] #Selecting needed data

    """
    #Getting the needed data from the event WITHOUT strucutre
    lappd_hits2 = hits2[hits2["detector_index"].isin(LAPPD_Indices)]
    counts2 = lappd_hits2.groupby(["event_id", "detector_index"]).size().reset_index(name="n_hits")
    pivoted2 = counts2.pivot(index="event_id", columns="detector_index", values="n_hits").fillna(0).astype(int)
    pivoted2.columns = [f"n_pmt{c}" for c in pivoted2.columns]

    tablePhotonsPerMuonHits2 = muons[["event_id", "pos_x", "pos_z","dir_x","dir_y","dir_z"]].merge(pivoted, on="event_id", how="left").fillna(0)
   
    data2 = tablePhotonsPerMuonHits2[["n_pmt132", "n_pmt133", "n_pmt134"]] #Selecting needed data


    hitRatio = pd.DataFrame()
    hitRatio[['ratio_pmt132','ratio_pmt133','ratio_pmt134']] = data[['n_pmt132','n_pmt133','n_pmt134']].div(data2[['n_pmt132','n_pmt133','n_pmt134']].values) #Gets the ratio by diving each column by its respective counterpart
    hitRatio = hitRatio.replace([float('inf'),-float('inf'),float('nan')],0) #Make sure all unreasonalbe values are set to a make of 2 to cap out like in the study
    """

    #Converting the data to numpy arrays
    data = data.to_numpy() #Selecting needed data and converting to np array
    """
    data2 = data2.to_numpy() #Selecting needed data and converting to np array
    hitRatio = hitRatio.to_numpy()
    """




    #Getting the x and z positions as well as the number of unique positions
    xpos = tablePhotonsPerMuonHits["pos_x"].to_numpy() #Converting x positions to a numpy array
    zpos = tablePhotonsPerMuonHits["pos_z"].to_numpy() #Converting z positions to a numpy array
    positionsArray = np.column_stack((xpos,zpos)) *(10**-3) #mm

    '''
    #Checking that the different versions of output.h5 have different bounds
    print(f"Max x position is {xpos.max()}")
    print(f"Min x position is {xpos.min()}")
    print(f"Max z position is {zpos.max()}")
    print(f"Min z position is {zpos.min()}")
    '''

    

    #Getting the x and z directions as well as the number of unique directions
    xDirec = tablePhotonsPerMuonHits["dir_x"].to_numpy() #Converting x directions to a numpy array
    zDirec = tablePhotonsPerMuonHits["dir_z"].to_numpy() #Converting z directions to a numpy array
    
    directionsArray = np.column_stack((xDirec,zDirec))

    #Getting unique coordinates and number of them
    uniqueCoords = np.unique(positionsArray, axis=0)
    uniquePosCount = len(uniqueCoords) #Number of unique coordinates

    #Getting unique x postions and sorting 
    uniqueX = np.unique(uniqueCoords[:,0])
    uniqueX.sort()

    #Getting unique directions and number of them
    uniqueDirec = np.unique(directionsArray, axis=0)
    uniqueDirecCount = len(uniqueDirec) #Number of unique coordinates

    numX = len(np.unique(xpos)) #gets the unique number of x cords | These are the columns 
    numZ = len(np.unique(zpos)) #gets the unique number of z cords | These are the rows

    #Check if there are any non zero directions, otherwise plot as normal
    if xDirec[0] != xDirec[1] or zDirec[0] != zDirec[1]:

        #Initialize circles
        circles_data= []

        #These will hold the hit data
        LAPPD_132_Forward = [] 
        LAPPD_133_Forward = [] 
        LAPPD_134_Forward = [] 

        #These will hold ratio data
        LAPPD_132_Forward_Ratio = [] 
        LAPPD_133_Forward_Ratio = [] 
        LAPPD_134_Forward_Ratio = [] 


        #Preliminary checks | consider this method https://stackoverflow.com/questions/2489435/check-if-a-number-is-a-perfect-square 
        direc_steps = np.sqrt(uniqueDirecCount) #square root of the number of unique directions present
        if direc_steps != int(direc_steps): #Check on this
            print('ERROR: DIRECTIONS ARE NOT PERFECT SQUARE')

        diameter = np.trunc((np.abs(uniqueX[0]-uniqueX[1])/direc_steps)*(10**3) * 0.85) #The end constant is just a size multiplier (units are mm)
        

        print(f"The spacing between unique x positions is: {(np.abs(uniqueX[0]-uniqueX[1]))*10**3}")
       

        #Get size of direction steps
        direc_step_size = np.abs(xDirec[0]-xDirec[uniqueDirecCount+1])
        print(f"The step size between each direction vector is: {direc_step_size}")

        print(f'The radius of each circle is: {diameter/2.0}')
        #Begin making the circles
        for i in range(len(tablePhotonsPerMuonHits)):
            shiftX = diameter * (xDirec[i]/direc_step_size)
            shiftZ = diameter * (zDirec[i]/direc_step_size)
            circ = patches.Circle(
                    (xpos[i]+shiftX, zpos[i]+shiftZ),
                    diameter/2.0)
                
            #Appending circle origins to list
            circles_data.append(circ)

            #Appending hit data to each list
            LAPPD_132_Forward.append(data[i,0])
            LAPPD_133_Forward.append(data[i,1])
            LAPPD_134_Forward.append(data[i,2])
            """
            #Appending ratio data to each list
            LAPPD_132_Forward_Ratio.append(hitRatio[i,0])
            LAPPD_133_Forward_Ratio.append(hitRatio[i,1])
            LAPPD_134_Forward_Ratio.append(hitRatio[i,2])
            """

        #Tick marks for graphing
        x_vals = np.linspace(np.min(xpos), np.max(xpos), numX) #m
        z_vals = np.linspace(np.min(zpos), np.max(zpos), numZ) #m

        #Getting the shape info into the proper form
        LAPPD_132_Forward = np.array(LAPPD_132_Forward)
        LAPPD_132_Forward_Ratio = np.array(LAPPD_132_Forward_Ratio)

        fig1, ax = plt.subplots(figsize=(20, 20)) 

        #Plotting LAPPD 132 Hits
        circ_collec = PatchCollection(circles_data,cmap='coolwarm',norm = colors.LogNorm(vmin =1,vmax = LAPPD_132_Forward.max()))
        circ_collec.set_array(LAPPD_132_Forward)
        ax.add_collection(circ_collec) 

        ax.set_xlabel("x position (mm)")
        ax.set_ylabel("y position (mm)") #Outward facing so it should be labeled "y"
        ax.set_xticks(x_vals)
        ax.set_yticks(z_vals)
        ax.set_title("LAPPD 132 Hits")
        ax.set_aspect("equal")
        ax.autoscale_view()

        '''
        #Plotting LAPPD 132 Hit Ratios
        #Uncomment this section and modify the section for the figure color bars to get the ratios
        fig2, ax2 = plt.subplots(figsize=(20, 20)) 

        circ_collec = PatchCollection(circles_data,cmap='coolwarm') # norm = colors(vmin =0,vmax=LAPPD_132_Forward_Ratio.max())
        circ_collec.set_array(LAPPD_132_Forward_Ratio)
        ax2.add_collection(circ_collec) 

        ax2.set_xlabel("x position (mm)")
        ax2.set_ylabel("y position (mm)") #Outward facing so it should be labeled "y"
        ax2.set_xticks(x_vals)
        ax2.set_yticks(z_vals)
        ax2.set_title("LAPPD 132 Hit Ratios")
        ax2.set_aspect("equal")
        ax2.autoscale_view()
        '''
        
        #LAPPD 133 Plotting
        fig2, ax2 = plt.subplots(figsize=(20, 20))
        LAPPD_133_Forward = np.array(LAPPD_133_Forward)
        LAPPD_133_Forward_Ratio = np.array(LAPPD_133_Forward_Ratio)
        circ_collec_133 = PatchCollection(circles_data,cmap='viridis',norm = colors.LogNorm(vmin =1,vmax = LAPPD_133_Forward.max()))
        circ_collec_133.set_array(LAPPD_133_Forward)
        ax2.add_collection(circ_collec_133) #This will be one color right now

        ax2.set_xlabel("x position (mm)")
        ax2.set_ylabel("y position (mm)") #Outward facing so it should be labeled "y"
        ax2.set_xticks(x_vals)
        ax2.set_yticks(z_vals)
        ax2.set_title("LAPPD 133 Hits")
        ax2.set_aspect("equal")
        ax2.autoscale_view()


        #LAPPD 134 Plotting
        fig3,ax3 = plt.subplots(figsize=(20,20))
        LAPPD_134_Forward = np.array(LAPPD_134_Forward)
        LAPPD_134_Forward_Ratio = np.array(LAPPD_134_Forward_Ratio)

        circ_collec_134 = PatchCollection(circles_data,cmap='coolwarm',norm = colors.LogNorm(vmin =1,vmax = LAPPD_134_Forward.max()))
        circ_collec_134.set_array(LAPPD_134_Forward)
        ax3.add_collection(circ_collec_134) #This will be one color right now

        ax3.set_xlabel("x position (mm)")
        ax3.set_ylabel("y position (mm)") #Outward facing so it should be labeled "y"
        ax3.set_xticks(x_vals)
        ax3.set_yticks(z_vals)
        ax3.set_title("LAPPD 134 Hits")
        ax3.set_aspect("equal")
        ax3.autoscale_view()

        
        #Parameters for all graphs
        fig1.colorbar(circ_collec,fraction = 0.05, ax=ax,location = 'right', label="hits")
        fig2.colorbar(circ_collec_133, fraction = 0.05 ,ax=ax2,location = 'right', label="hits")
        fig3.colorbar(circ_collec_134, fraction = 0.05, ax=ax3,location = 'right', label="hits")
       
        plt.show()      
    else:

        #NOTE: The below section will be for code with no angle put in
        #Plotting color mesh
        x_vals = np.linspace(np.min(xpos), np.max(xpos), numX)/(10**3) #m
        z_vals = np.linspace(np.min(zpos), np.max(zpos), numZ)/(10**3) #m
        XX,ZZ = np.meshgrid(x_vals, z_vals)
    
    
        #Plotting LAPPD 132 Hits
        fig,ax = plt.subplots()
        plt.pcolormesh(XX,ZZ,data[:, 0].reshape(numZ,numX), cmap='viridis', shading='auto',edgecolors = 'r',linewidths=0.5,norm = colors.LogNorm(vmin=1, vmax=data[:, 0].max()))
        plt.title("LAPPD 132 Hits from Muon Vertex Positions")
        ax.set_xticks(x_vals)
        ax.set_yticks(z_vals)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)") #this is outward facing so it should be labeled as y
        cbar1 = plt.colorbar()
        cbar1.set_label('Number of Hits', fontsize=12, rotation=270, labelpad=15)
    
        plt.show()


        #Plotting LAPPD 133 Hits
        fig2,ax2 = plt.subplots()
        plt.pcolormesh(XX,ZZ,data[:, 1].reshape(numZ,numX), cmap='viridis', shading='auto',edgecolors = 'r',linewidths=0.5,norm = colors.LogNorm(vmin=1, vmax=data[:, 1].max()))
        plt.title("LAPPD 133 Hits from Muon Vertex Positions")
        ax2.set_xticks(x_vals)
        ax2.set_yticks(z_vals)
        ax2.set_xlabel("X Position (m)")
        ax2.set_ylabel("Y Position (m)")
        cbar2 = plt.colorbar()
        cbar2.set_label('Number of Hits', fontsize=12, rotation=270, labelpad=15)

        plt.show()

        #Plotting LAPPD 134 Hits
        fig3,ax3 = plt.subplots()
        plt.pcolormesh(XX,ZZ,data[:, 2].reshape(numZ,numX), cmap='viridis', shading='auto',edgecolors = 'r',linewidths=0.5,norm = colors.LogNorm(vmin=1, vmax=data[:, 2].max()))
        plt.title("LAPPD 134 Hits from Muon Vertex Positions")
        ax3.set_xticks(x_vals)
        ax3.set_yticks(z_vals)
        ax3.set_xlabel("X Position (m)")
        ax3.set_ylabel("Y Position (m)")
        cbar3 = plt.colorbar()
        cbar3.set_label('Number of Hits', fontsize=12, rotation=270, labelpad=15)
        plt.show()
if __name__ == "__main__":
    main()
