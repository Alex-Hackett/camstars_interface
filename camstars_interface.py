#!/opt/ioa/software/anaconda/anaconda36/envs/20190218_py36/bin/python
# -*- coding: utf-8 -*-
"""
Plot Reader reads in the data stored in Cambridge STARS plot files, which
detail a single run of the stellar evolution code
"""

#Imports
import os
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import astropy as ap
from astropy.io import ascii as astro_ascii

class dataStore:
    def __init__(self, filename='fullPlot'):
        self.filename = filename
        self.readPlot()
        self.extractCols()
        
    def readPlot(self):
        self.full_table = astro_ascii.read(self.filename)
        
    def extractCols(self):
        self.num = self.full_table.columns[0]
        self.age = self.full_table.columns[1]
        self.log_rad = self.full_table.columns[2]
        self.log_teff = self.full_table.columns[3]
        self.log_lum = self.full_table.columns[4]
        self.mass = self.full_table.columns[5]
        self.He_core_mass = self.full_table.columns[6]
        self.CO_core_mass = self.full_table.columns[7]
        self.log_H_lum = self.full_table.columns[8]
        self.log_He_lum = self.full_table.columns[9]
        self.log_C_lum = self.full_table.columns[10]
        self.conv_boundary = self.full_table.columns[11:22]
        self.max_H_eng = self.full_table.columns[23]
        self.max_He_eng = self.full_table.columns[24]
        self.log_opacity = self.full_table.columns[25]
        self.timestep = self.full_table.columns[26]
        self.surf_H = self.full_table.columns[27]
        self.surf_He = self.full_table.columns[28]
        self.surf_C = self.full_table.columns[29]
        self.surf_N = self.full_table.columns[30]
        self.surf_O = self.full_table.columns[31]
        self.surf_He_3 = self.full_table.columns[32]
        self.roche_lobe_frac = self.full_table.columns[33]
        self.spin_AM = self.full_table.columns[34]
        self.binary_period = self.full_table.columns[35]
        self.binary_seperation = self.full_table.columns[36]
        self.binary_mass = self.full_table.columns[37]
        self.orbital_AM = self.full_table.columns[38]
        self.total_spin_AM = self.full_table.columns[39]
        self.total_AM = self.full_table.columns[40]
        self.orb_ang_freq = self.full_table.columns[41]
        self.star_ang_freq = self.full_table.columns[42]
        self.star_mom_inert = self.full_table.columns[43]
        self.orb_mom_inert = self.full_table.columns[44]
        self.mass_loss_rate = self.full_table.columns[45]
        self.burn_shell_boundary = self.full_table.columns[46:57]
        self.thermohaline_mix_boundary = self.full_table.columns[58:69]
        self.convec_env_mass = self.full_table.columns[70]
        self.radius_convec_env = self.full_table.columns[71]
        self.log_rho_c = self.full_table.columns[72]
        self.log_T_c = self.full_table.columns[73]
        
    def HRD(self, cut_PMS=True):
        if cut_PMS:
            plt.plot(self.log_teff[np.where(self.He_core_mass > 1e-5 * self.mass[0])], self.log_lum[np.where(self.He_core_mass > 1e-5 * self.mass[0])])
            plt.xlim(max(self.log_teff) ,min(self.log_teff))
            plt.title(r'HRD')
            plt.xlabel(r'Effective Temperature $log_{10}(K)$')
            plt.ylabel(r'Luminosity $log_{10}(L/L_{\odot})$')
            plt.show()
        else:
            plt.plot(self.log_teff, self.log_lum)
            plt.xlim(max(self.log_teff) ,min(self.log_teff))
            plt.title(r'HRD')
            plt.xlabel(r'Effective Temperature $log_{10}(K)$')
            plt.ylabel(r'Luminosity $log_{10}(L/L_{\odot})$')
            plt.show()
        
    def rhoCTC(self):
        plt.plot(self.log_rho_c, self.log_T_c)
        plt.title('Central Conditions')
        plt.xlabel(r'Central Density $log_{10}(g/cm^{3})$')
        plt.ylabel(r'Central Temperature $log_{10}(K)$')
        plt.show()
        
    def evolCore(self):
        plt.plot(self.age, self.He_core_mass, 'g', label = 'Helium Core Mass')
        plt.plot(self.age, self.CO_core_mass, 'b', label = 'CO Core Mass')
        plt.xlabel('Model Age (Yrs)')
        plt.ylabel('Evolved Core Mass')
        plt.legend()
        plt.show()
        
    def kippDiag(self, lttc=True):
        if lttc:
            age = np.log10(max(self.age) - self.age)
        else:
            age = self.age
        for i in range(len(self.conv_boundary)):
            if self.conv_boundary[i].any() > 0:
                plt.plot(age, self.conv_boundary[i], 'black')
        plt.ylim(0, max(self.mass))
        plt.plot(age, self.max_H_eng, 'r', label = 'Max H Energy Generation')
        plt.plot(age, self.max_He_eng, 'g', label = 'Max He Energy Generation')
        plt.legend()
        plt.title('Kippenhahn Diagram')
        if lttc:
            plt.xlabel('Log Time to Collapse ($log_{10}(yrs)$)')
            plt.xlim(max(age), min(age[age != -np.inf]))
        else:
            plt.xlabel('Model Age (Yrs)')
        plt.ylabel('Mass Coordinate ($M_{\odot}$)')
        plt.show()
        
    def burnShell(self, lttc=True):
        if lttc:
            age = np.log10(max(self.age) - self.age)
        else:
            age = self.age
        for i in range(len(self.burn_shell_boundary)):
            if self.burn_shell_boundary[i].any() > 0:
                plt.plot(age, self.burn_shell_boundary[i], 'black')
        plt.plot(age, self.max_H_eng, 'r', label = 'Max H Energy Generation')
        plt.plot(age, self.max_He_eng, 'g', label = 'Max He Energy Generation')
        plt.title('Burning Shells Diagram')
        if lttc:
            plt.xlabel('Log Time to Collapse ($log_{10}(yrs)$)')
            plt.xlim(max(age), min(age[age != -np.inf]))
        else:
            plt.xlabel('Model Age (Yrs)')
        plt.ylabel('Mass Coordinate ($M_{\odot}$)')
        plt.legend()
        plt.show()
        
    def lumTrack(self):
        plt.plot(self.age, self.log_H_lum, 'black', label = 'Hydrogen')
        plt.plot(self.age, self.log_He_lum, 'green', label = 'Helium')
        plt.plot(self.age, self.log_C_lum, 'blue', label = 'Carbon')
        plt.legend()
        plt.title('Luminosity Sources with Time')
        plt.xlabel('Model Age (Yrs)')
        plt.ylabel('Luminosity $log_{10}(L/L_{\odot})$')
        plt.show()
        
    def convecMass(self):
        plt.plot(self.age, self.convec_env_mass)
        plt.title('Mass in Convective Envelope')
        plt.xlabel('Model Age (Yrs)')
        plt.ylabel('Mass ($M/M_{\odot}$)')
        plt.show()
        
    def convecRadius(self):
        plt.plot(self.age, self.radius_convec_env)
        plt.title('Radius of Base of Convective Envelope')
        plt.xlabel('Model Age (Yrs)')
        plt.ylabel('Radial Coordinate ($R/R_{\odot}$)')
        plt.show()
        
    def radiusWithTime(self):
        plt.plot(self.age, 10**self.log_rad)
        plt.title('Model Radius with Time')
        plt.xlabel('Model Age (Yrs)')
        plt.ylabel('Model Radius ($R/R_{\odot}$)')
        plt.show()


class runUtils:
    #def __init__(self):#, workdir):
        

        
    def copyInputs(self):

        os.system('rm modin')
        os.system('tail -399 modout > modin')
        
        os.system('rm nucmodin')
        os.system('mv nucmodout nucmodin')
            
    def updatePlot(self):
        plotorg = open('plot', 'r')
        if not os.path.isfile('fullPlot'):
            fullPlot = open('fullPlot', 'w+')
            fullPlot.close()
        
        fullPlot = open('fullPlot', 'r')
        
        plotorgData = plotorg.read()
        fullPlotData = fullPlot.read()
    
        
        newFullPlotData = fullPlotData + plotorgData
        fullPlot.close()
        fullPlot = open('fullPlot', 'w')
        
        fullPlot.write(newFullPlotData)
        fullPlot.close()
            
    def runSim(self):
        os.system('./run_bs')
        
    def compiler(self):
        os.system('make')
        
        
class inputManipulation:
    #1.000000E+00 is the desired input format
    #9999 is the input format for timesteps
    #199 is the input format for mesh points
    def __init__(self):#, workdir):
        self.modinRead()
        self.dataRead()

    def modinRead(self):
        #self.modin = open('modin', 'r+')
        line_stream = open('modin').readline().rstrip()
        line_stream = line_stream.split()
        self.modin_mass = line_stream[0]
        self.modin_timestep_size = line_stream[1]
        self.modin_age = line_stream[2]
        self.modin_bin_period = line_stream[3]
        self.modin_total_bin_mass = line_stream[4]
        self.modin_art_eng_gen = line_stream[5]
        self.modin_mesh_point_num = line_stream[6]
        self.modin_desired_num_models = line_stream[7]
        self.modin_star_model_num = line_stream[8]
        self.modin_which_star = line_stream[9]
        try:
            self.modin_H_shell_pressure = line_stream[10]
            self.modin_He_shell_pressure = line_stream[11]
        except IndexError:
            print('Pressure Data Missing')
            self.modin_H_shell_pressure = ''
            self.modin_He_shell_pressure = ''
            
    def modinReWrite(self):
        new_line_stream = '   '+self.modin_mass+ '  ' +self.modin_timestep_size+ '  ' +self.modin_age+ '  ' +self.modin_bin_period+ '  ' +self.modin_total_bin_mass+ '  ' +self.modin_art_eng_gen+ '  ' +self.modin_mesh_point_num+ ' '+self.modin_desired_num_models+ '  ' +self.modin_star_model_num+ '    ' +self.modin_which_star+'\n'#+ ' '+self.modin_H_shell_pressure+ ' '+ self.modin_He_shell_pressure+'\n'

        with open("modin") as f:
            lines = f.readlines()
        lines[0] = new_line_stream
        lines = "".join(lines)
        with open('modin', 'w') as f:
            for item in lines:
                f.write(item)
                
    def modinClose(self):
        self.modin.close()
        
        
        
    def dataRead(self):
        line_stream = open('data').readline().rstrip()
        line_stream = line_stream.split()
        self.data_mesh_points = line_stream[0]
        self.data_first_timestep_max_iter  = line_stream[1]
        self.data_later_timestep_max_iter = line_stream[2]
        self.data_jin = line_stream[3]
        self.data_jout = line_stream[4]
        self.data_how_remesh = line_stream[5]
        self.data_last_corrections = line_stream[6]
        self.data_thermal_gen_rate = line_stream[7]
        self.data_H_burn = line_stream[8]
        self.data_He_burn = line_stream[9]
        self.data_C_burn = line_stream[10]
        self.data_binary_mode = line_stream[11]
        
    def dataReWrite(self):
        new_line_stream = ' '+self.data_mesh_points+'  '+self.data_first_timestep_max_iter+'  '+self.data_later_timestep_max_iter+'  '+self.data_jin+'  '+self.data_jout+'   '+self.data_how_remesh+'   '+self.data_last_corrections+'   '+self.data_thermal_gen_rate+'   '+self.data_H_burn+'   '+self.data_He_burn+'   '+self.data_C_burn+'   '+self.data_binary_mode+'\n'
        
        with open('data') as f:
            lines = f.readlines()
        lines[0] = new_line_stream
        lines = "".join(lines)
        with open('data', 'w') as f:
            for item in lines:
                f.write(item)
                
                
#'''
#Testing, to automatically contract a star of 1 solar mass to the ZAMS, then switch on nuke 
#burning, and burn until the He Flash
#God willing this actually works...
#'''
#if not os.path.isfile('modin'):
#    os.system('tar -xvf bs2007.tar.gz')
#
##Set these classes up now
#run = runUtils()
#inputs = inputManipulation()
#
##Compile the code
#run.compiler()
#
##Alter the mass of the star to 1 solar mass
#inputs.modin_mass = '1.000000E+00'
##Run for 2000 timesteps to ensure we reach the ZAMS
#inputs.modin_desired_num_models = '2000'
##Shut off the hydrogen, helium and carbon burning
#inputs.data_H_burn = '0'
#inputs.data_He_burn = '0'
#inputs.data_C_burn = '0'
##Write to data and modin
#inputs.modinReWrite()
#inputs.dataReWrite()
#
##Run the model
#run.runSim()
#
##Update the plots
#run.updatePlot()
#
##Update the modin files 
#run.copyInputs()
#
##Read in the new modin and data files
#inputs.modinRead()
#inputs.dataRead()
#
##Run 9999 timessteps now, to reach He Flash
#inputs.modin_desired_num_models = '9999'
##Turn on Hydrogen, Helium and Carbon burning
#inputs.data_H_burn = '1'
#inputs.data_He_burn = '1'
#inputs.data_C_burn = '1'
#
##Write data and modin
#inputs.modinReWrite()
#inputs.dataReWrite()
#
##Run the model again
#run.runSim()
#
##Update the plotfile
#run.updatePlot()
#
##Generate the plotting object
#plot = dataStore()
##Plot an HRD
#plot.HRD(cut_PMS=False)
        
    
        
            
        
        
        
