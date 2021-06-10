#Charlie Meisel
#Python Functions for use in XRD plotting and EIS data formatting
#September 14, 2020.  Code written earlier in Jupytr Notebooks

"If a plotting program has an 's' at the end of it, it is ment to plot multiple files in one plot"
"The Half Cell Arrhenius (AH) plots are made specifically for how I store the AH data"

#Imports
import os
import sys
from shutil import copyfile
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io.parsers import ParserError #to be able to identify the parser error
import numpy as np
from sklearn.linear_model import LinearRegression #Will help with plotting linear regression line
from scipy import stats
import scipy as scipy
import csv
import time
import datetime
import scipy as sp

'Bayes DRT import'
from bayes_drt import inversion #Recommended by Jake before tutorial
from bayes_drt.inversion import Inverter#inverter class in inversion module
from bayes_drt import eis_utils as gt #throwback to gamry tools
from bayes_drt.stan_models import save_pickle,load_pickle #useful for saving a fit for later (saves whole inverter object)

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"XRD functions"
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def xrd_format(location): #Used on a CSV from the XRD machine to format the data for plotting and convert intensity to relative intensity.  The input is a string
    try: #added this because sometimes the data starts a row or two lower
        df = pd.read_csv(location,skiprows=26) #creates datafile from csv convert of XRDML file
        maximum = df['Intensity'].max() #calculates highes intensity value
    except KeyError:
        try:
            df = pd.read_csv(location,skiprows=27)
            maximum = df['Intensity'].max()
        except KeyError:
            df = pd.read_csv(location,skiprows=28)
            maximum = df['Intensity'].max() 
    df['Intensity'] = df['Intensity']/maximum #
    df = df.rename({'Angle':'a','Intensity':'i'},axis=1) #renames columns to make further plotting quicker a = angle i = relative intensity
    return df

def xrd_format_icdd(sheet): #Returns 2Theta and relative intensity data from my saved ICDD files. The input is the name of the sheet (material name)
    df = pd.read_excel('/Users/Charlie/Documents/CSM/XRD_Data/ICDD_XRD_Files.xlsx',sheet) #will need to change this line if the file location changes
    df = df[['2Theta','Relative Intensity']]
    df = df.rename({'2Theta':'a','Relative Intensity':'i'},axis=1)
    return df

def plot_xrd(loc,material): #this function graphs the XRD spectra from a CSV. Both inputs are strings. Loc is the location of the file. material is what the line will be named
    df = xrd_format(loc)
    plt.figure(dpi=250)  #change to change the quality of the return chart
    plt.plot(df['a'],df['i'], label = material)
    plt.xlim(20,80)
    plt.xlabel('2\u03B8')
    plt.ylabel('Relative Intensity')
    plt.legend()

def plot_xrds(loc, material,y_offset=0): #This function enables multiple spectra to be on the same plot. this function graphs the XRD spectra from a CSV. loc and material inputs are strings, while y_offset is the y offset and if left blank defaults to 0. Loc is the location of the file. material is what the line will be named
    try: #loc should be the sheet name in the ICDD file or the location of the csv file
        df = xrd_format_icdd(loc) 
    except:
        df = xrd_format(loc)
    plt.plot(df['a'],df['i']+y_offset, label = material) #offset is the y axis offset to stack graphs and is optional
    plt.xlabel('2\u03B8')
    plt.ylabel('Relative Intensity')
    plt.legend()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"Gamry EIS data formatting functions - still needs some testing"
"I need to test to see if the right CSV is made then saved in the right place"
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def dta2csv(loc): #input is file location, this just creats an identical file and changes its format from DTA to CSV
    file = loc
    copyfile(file, file.replace('.DTA','')+'.csv')
    
def iv_data(area,loc): #Takes DTA file and returns a CSV with the desired data using dataframes - not tested
    "Converts and finds CSV then turns it into a dataframe"
    dta2csv(loc)
    loc_csv = loc.replace('.DTA','')+'.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here, but it works
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    "calculations and only keeping the useful data"
    df['A'] = df['A'].div(-area)
    df['W'] = df['W'].div(-area)
    df_useful = df[['V','A','W']]
    "finds maximum values"
    #max_w = df_useful['W'].max()
    #max_v = df.loc[df.index[df['W'] == max_w],'V'].item()
    #print('Max Power Density:',round(max_w,3),'W/cm^2 at',max_v,'V') #prints maximum values
    #df_useful.to_csv(loc_csv) #Creates a CSV file with the desired data: voltage, current, and power
    return df_useful   
    
def ocv_data(loc): #Returns time and voltage from a OCV test in a csv using dataframes - not tested
    dta2csv(loc)
    loc_csv = loc.replace('.DTA','')+'.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
    skip = 0
    for row in file: #searches first column of each row in csv for "CURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            print(skip)
            break
        if row[0] == 'READ VOLTAGE': #For whatever reason the DTA files are different if the data is aborted
            skip = file.line_num+1
            print(skip)
            break
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    df_useful = df[['s','V']]
    df_useful.to_csv(loc_csv)

def peis_data(area,loc): #Returns Zreal and Zimag from a DTA file of PotentiostaticEis in a CSV - not tested
    dta2csv(loc) #convert DTA to a CSV
    loc_csv = loc.replace('.DTA','')+'.csv' #access newly made file
    #find right amount of rows to skip
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'ZCURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1')
    df['ohm.1'] = df['ohm.1'].mul(-1*area)
    df['ohm'] = df['ohm'].mul(area)
    df_useful = df[['ohm','ohm.1']]
    df_useful.to_csv(loc_csv)

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"Gamry EIS data plotting functions - still need some work"
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

def plot_ocv(loc): #Plots OCV vs time from a DTA file
    dta2csv(loc)
    loc_csv = loc.replace('.DTA','')+'.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
    skip = 0
    for row in file: #searches first column of each row in csv for "CURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
        if row[0] == 'READ VOLTAGE': #For whatever reason the DTA files are different if the data is aborted
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    df_useful = df[['s','V']]
    plot = plt.figure()
    plt.plot(df_useful['s'],df_useful['V'],'ko',)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    return plot

def plot_peis(area,loc): #Plots Zreal and Zimag from a DTA file of PotentiostaticEis
    dta2csv(loc) #convert DTA to a CSV
    loc_csv = loc.replace('.DTA','')+'.csv' #access newly made file
    #find right amount of rows to skip
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'ZCURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1')
    df['ohm.1'] = df['ohm.1'].mul(-1*area)
    df['ohm'] = df['ohm'].mul(area)
    df_useful = df[['ohm','ohm.1']]
    #Plotting
    plot = plt.figure()
    plt.plot(df_useful['ohm'],df_useful['ohm.1'],'o',color = '#21314D') # #21314D is the color of Mines Navy blue
    plt.xlabel('Zreal (\u03A9*cm$^2$)')
    plt.ylabel('-Zimag (\u03A9*cm$^2$)')
    plt.axhline(y=0, color='#D2492A', linestyle='-.') # #D2492A is the color of Mines orange
    plt.rc('font', size=16)
    plt.axis('scaled')
    return plot

def plot_peiss(area, condition, loc): #Enables multiple EIS spectra to be stacked on the same plot
    dta2csv(loc) #convert DTA to a CSV
    loc_csv = loc.replace('.DTA','')+'.csv' #access newly made file
    #find right amount of rows to skip
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'ZCURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1')
    df['ohm.1'] = df['ohm.1'].mul(-1*area)
    df['ohm'] = df['ohm'].mul(area)
    df_useful = df[['ohm','ohm.1']] #returns the useful information
    plt.plot(df_useful['ohm'],df_useful['ohm.1'],'o',label = condition,) #plots data
    plt.xlabel('Zreal (\u03A9*cm$^2$)')
    plt.ylabel('-Zimag (\u03A9*cm$^2$)')
    plt.axhline(y=0,color='k', linestyle='-.') # plots line at 0 #D2492A is the color of Mines orange
    plt.rc('font', size=12)
    plt.axis('scaled') #Keeps X and Y axis scaled 1 to 1
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    #plt.legend(loc='best')
    plt.tight_layout()

def plot_geis(loc): #Returns a plot of time vs voltage for Galvanostatic EIS 
    'Add to this Area and print the current density on the plot'
    dta2csv(loc) #convert DTA to a CSV
    loc_csv = loc.replace('.DTA','')+'.csv' #access newly made file
    #find right amount of rows to skip and making useful dataframe
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1')
    df_useful = df[['s','V']]
    #Plotting
    plot = plt.figure()
    plt.plot(df_useful['s'],df_useful['V'],'o',color = '#21314D') # #21314D is the color of Mines Navy blue
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.rc('font', size=16)
    return plot

def plot_ivfc(area, loc): #plots IV curve and power density curve with the Pmax listed on the plot 
    dta2csv(loc) #Converts and finds CSV then turns it into a dataframe
    loc_csv = loc.replace('.DTA','')+'.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here, but it works
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    #calculations and only keeping the useful data
    df['A'] = df['A'].div(-area)
    df['W'] = df['W'].div(-area)
    df_useful = df[['V','A','W']]
    #Plotting
    fig, ax1 = plt.subplots()
    #IV plotting
    color = '#21314D' #Navy color
    ax1.set_xlabel('Current Density ($A/cm^2$)')
    ax1.set_ylabel('Voltage (V)', color=color)
    ax1.plot(df_useful['A'], df_useful['V'],'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    #Power density plotting
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = '#D2492A' #orange color
    ax2.set_ylabel('Power Density ($W/cm^2$)', color=color2)  # we already handled the x-label with ax1
    ax2.plot(df_useful['A'], df_useful['W'], 'o',color=color2) 
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #Calculating and printing max values onto the graph
    max_w = df_useful['W'].max() #finds maximum power density
    max_v = df.loc[df.index[df['W'] == max_w],'V'].item() #finds voltage of max power density
    max_ws = f'{round(max_w,3)}' #setts float to a string
    max_vs = f'{round(max_v,3)}'
    plt.figtext(0.28,0.21,r'$P_{max} = $'+max_ws+r' $W/cm^2 at$ '+max_vs+r'$V$',size='large',weight='bold')
    #plt.title(r'$P_{max}$'+max_ws+r'W/cm$^2$ at')#+max_vs+'V')

    dta2csv(loc) #Converts and finds CSV then turns it into a dataframe
    loc_csv = loc.replace('.DTA','')+'.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here, but it works
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    #calculations and only keeping the useful data
    df['A'] = df['A'].div(-area)
    df['W'] = df['W'].div(-area)
    df_useful = df[['V','A','W']]
    #Plotting
    fig, ax1 = plt.subplots()
    #IV plotting
    ax1.set_xlabel('Current Density ($A/cm^2$)')
    ax1.set_ylabel('Voltage (V)')
    ax1.plot(df_useful['A'], df_useful['V'],'o')
    ax1.tick_params(axis='y')
    #Power density plotting
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Power Density ($W/cm^2$)')  # we already handled the x-label with ax1
    ax2.plot(df_useful['A'], df_useful['W'], 'o')
    ax2.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

def plot_ivec(area, loc): #plots IV curve in EC mode and displays the current density at 1.5V on the plot
    dta2csv(loc) #Converts and finds CSV then turns it into a dataframe
    loc_csv = loc.replace('.DTA','')+'.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here, but it works
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    'calculations and only keeping the useful data'
    df['A'] = df['A'].div(area)
    df_useful = df[['V','A']]
    'Plotting'
    fig, ax1 = plt.subplots()
    'IV plotting'
    color = '#21314D'
    ax1.set_xlabel('Current Density ($A/cm^2$)')
    ax1.set_ylabel('Voltage (V)', color=color)
    ax1.plot(df_useful['A'], df_useful['V'],'o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    #plt.axhline(y=1.28, color='#D2492A', linestyle='-.') # #D2492A is the color of Mines orange, 1.28 is the thermoneutral voltage
    'Calculating and printing current density at 1.5V'
    current_density15 = -df_useful[abs(df_useful['V'])>=1.49].iloc[0,1] #finds the current density of the first Voltage value above 1.5V
    V15 = df_useful[abs(df_useful['V'])>=1.49].iloc[0,0] #same as before but returns the exact voltage value
    current_density15_string = f'{round(current_density15,3)}'
    V15_string = f'{round(-V15,3)}'
    plt.figtext(0.28,0.21,current_density15_string+r'$A/cm^2\:at$ '+V15_string+r'$V$',size='large',weight='bold') #placing value on graph

def plot_ivecs(area,condition,loc): #plots multiple EC IV curves on same plot
    dta2csv(loc) #Converts and finds CSV then turns it into a dataframe
    loc_csv = loc.replace('.DTA','')+'.csv'
    file = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here, but it works
    for row in file: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
        if row[0] == 'CURVE':
            skip = file.line_num+1
            break
    df = pd.read_csv(loc_csv,sep='\t',skiprows=skip,encoding='latin1')
    'calculations and only keeping the useful data'
    df['A'] = df['A'].div(area)
    df_useful = df[['V','A']]
    'Plotting'
    plt.plot(df_useful['A'], df_useful['V'],'o', label=condition)
    plt.xlabel('Current Density ($A/cm^2$)')
    plt.ylabel('Voltage (V)')
    #ax1.tick_params(axis='y')
    plt.legend(loc='best')
    plt.tight_layout()

def plot_galvanoDeg(FolderLoc,fit = 'yes',smooth='no',first_file = 'default'): #If you want to change the first file, put the loc in place of 'default
    files = os.listdir(FolderLoc)
    useful_files = [] #initializing a list for the useful files
    #Taking out all of the galvanostatic files
    for file in files: #looping over all files in the folder Folderloc
        if (file.find('GS')!=-1) and (file.find('.DTA')!=-1):
            #Extracting the file number (used for sorting, yeah this is probably a roundabout way of doing it)
            start, end = file.rsplit('#',1)# cutting everything out before the # so only the number and file extension is left
            fnum, fileExt = end.rsplit('.',1) #cutting off the file extension leaving only the number as a string
            index = int(fnum) #Converts the file number to a string
            useful_file = (file,index)
            useful_files.append(useful_file)
    #Sorting the files
    useful_files.sort(key=lambda x:x[1]) #Sorts by the second number in the tupple
    sorted_useful_files, numbers = zip(*useful_files) #splits the tupple
    sorted_useful_files = [FolderLoc + '/' + f for f in sorted_useful_files] #Turning all files from their relative paths to the absolute path
    #Getting the first time
    if first_file == 'default':
        T0_stamp = gt.get_timestamp(sorted_useful_files[0]) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Conveting Datetime to seconds from Epoch
    else:
        T0_stamp = gt.get_timestamp(first_file) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Conveting Datetime to seconds from Epoch
    #Combining all dataframes
    dfs = [] #Initializing list of dfs
    length = len(sorted_useful_files) #gets length of sorted useful files
    for i in range(0,length,1):
        loc = os.path.join(FolderLoc,sorted_useful_files[i]) #Creats a file path to the file of choice
        dta2csv(loc) #convert DTA to a CSV
        loc_csv = loc.replace('.DTA','')+'.csv' #access newly made file
        data = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
        for row in data: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
            if row[0] == 'CURVE':
                skip = data.line_num+1
                break
        df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1') #creat data frame for a file
        start_time = gt.get_timestamp(sorted_useful_files[i]).strftime("%s") #Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','V vs. Ref.']]
        dfs.append(df_useful)
    cat_dfs = pd.concat(dfs,ignore_index=True)# (s) Combine all the dataframes in the file folder
    cat_dfs.sort_values(by=['s'])
    cat_dfs['s'] = (cat_dfs['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    #plotting:
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (hrs)')
    ax.set_ylabel('Voltage (V)')
    #ax.plot(cat_dfs['s'],cat_dfs['V vs. Ref.'],'.k',linewidth=5)
    if smooth == 'no':
        ax.plot(cat_dfs['s'],cat_dfs['V vs. Ref.'],'.k')
    if smooth == 'yes':
        bin_size = 50
        bins = cat_dfs['V vs. Ref.'].rolling(bin_size)
        moving_avg_voltage = bins.mean()
        ax.plot(cat_dfs['s'],moving_avg_voltage,'k')
    ax.set_ylim(0,1.2)
    #fitting and writting slope on graph:
    if fit == 'yes':
        m,b = np.polyfit(cat_dfs['s'],cat_dfs['V vs. Ref.'],1)
        fit = m*cat_dfs['s']+b
        ax.plot(cat_dfs['s'],fit,'--r')
        mp = m*-100000 #Converting the slope into a % per khrs (*100 to get to %, *1000 to get to khrs,*-1 for degredation)
        ms = f'{round(mp,3)}'
        plt.figtext(0.31,0.15,'Degredation: '+ms+'% /khrs',weight='bold')
    plt.show()

def plot_ocvDeg(FolderLoc,fit='yes',first_file = 'default'):
    files = os.listdir(FolderLoc)
    useful_files = [] #initializing a list for the useful files
    #Taking out all of the galvanostatic files
    for file in files: #looping over all files in the folder Folderloc
        if (file.find('OCV')!=-1) and (file.find('.DTA')!=-1) and (file.find('Deg')!=-1):
            useful_files.append(os.path.join(FolderLoc,file))
    for file in useful_files: #Finding the first file
        if file.find('Deg__#1')!=-1:
            file1 = os.path.join(FolderLoc,file)
    if first_file == 'default': #if another file is specified as the first file, this file will be used to find T0
        T0_stamp = gt.get_timestamp(file1) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Conveting Datetime to seconds from Epoch
    else:
        T0_stamp = gt.get_timestamp(first_file) #gets time stamp from first file
        t0 = T0_stamp.strftime("%s") #Conveting Datetime to seconds from Epoch
    dfs = [] #Initializing list of dfs
    length = len(useful_files) #gets length of sorted useful files
    for i in range(0,length,1):
        dta2csv(useful_files[i]) #convert DTA to a CSV
        loc_csv = useful_files[i].replace('.DTA','')+'.csv' #access newly made file
        data = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
        for row in data: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
            if row[0] == 'CURVE':
                skip = data.line_num+1
                break
        df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1') #creat data frame for a file
        start_time = gt.get_timestamp(useful_files[i]).strftime("%s") #Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','V vs. Ref.']]
        dfs.append(df_useful)
    cat_dfs = pd.concat(dfs)# (s) Combine all the dataframes in the file folder
    cat_dfs['s'] = (cat_dfs['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (hrs)')
    ax.set_ylabel('Voltage (V)')
    ax.plot(cat_dfs['s'],cat_dfs['V vs. Ref.'],'.k')
    ax.set_ylim(0,1.2)
    #fitting and writting slope on graph: 
    if fit == 'yes':
        m,b = np.polyfit(cat_dfs['s'],cat_dfs['V vs. Ref.'],1)
        fit = m*cat_dfs['s']+b
        ax.plot(cat_dfs['s'],fit,'--r')
        mp = m*-100000 #Converting the slope into a % per khrs (*100 to get to %, *1000 to get to khrs,*-1 for degredation)
        ms = f'{round(mp,3)}'
        plt.figtext(0.31,0.15,'Degredation: '+ms+'% /khrs',weight='bold')
    plt.show()

def plot_potentioDeg(area,FolderLoc,fit = 'yes'): #Doesnt quite work
    files = os.listdir(FolderLoc)
    useful_files = []
    #Taking out all of the galvanostatic files
    for file in files:
        if (file.find('PS_')!=-1) and (file.find('.DTA')!=-1):
            #Extracting the file number (used for sorting, yeah this is probably a roundabout way of doing it)
            start, end = file.rsplit('#',1)# cutting everything out before the # so only the number and file extension is left
            fnum, fileExt = end.rsplit('.',1) #cutting off the file extension leaving only the number as a string
            index = int(fnum) #Converts the file number to a string
            if (file.find('Deg10')!=-1):
                index += 10
            useful_file = (file,index)
            useful_files.append(useful_file)
    #Sorting the files
    useful_files.sort(key=lambda x:x[1]) #Sorts by the second number in the tupple
    sorted_useful_files, numbers = zip(*useful_files) #splits the tupple
    #Getting the first time
    sorted_useful_files = [FolderLoc + '/'+f for f in sorted_useful_files]
    T0_stamp = gt.get_timestamp(sorted_useful_files[0]) #gets time stamp from first file
    t0 = T0_stamp.strftime("%s") #Conveting Datetime to seconds from Epoch
    #Combining all dataframes
    dfs = [] #Initializing list of dfs
    len = len(sorted_useful_files) #gets length of sorted useful files
    for i in range(0,len,1):
        loc = os.path.join(FolderLoc,sorted_useful_files[i]) #Creats a file path to the file of choice
        dta2csv(loc) #convert DTA to a CSV
        loc_csv = loc.replace('.DTA','')+'.csv' #access newly made file
        data = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
        for row in data: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
            if row[0] == 'CURVE':
                skip = data.line_num+1
                break
        df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1') #creat data frame for a file
        start_time = gt.get_timestamp(sorted_useful_files[i]).strftime("%s") #Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','A']]
        dfs.append(df_useful)
    cat_dfs = pd.concat(dfs)# (s) Combine all the dataframes in the file folder
    cat_dfs['s'] = (cat_dfs['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    cat_dfs['A'] = cat_dfs['A'].div(area)
    #plotting:
    fig,ax = plt.subplots()
    ax.set_xlabel('Time (hrs)')
    ax.set_ylabel('Current Density (A/cm$^2$)')
    ax.plot(cat_dfs['s'],-cat_dfs['A'],'k')
    #ax.set_ylim(0,1.2)
    #fitting and writting slope on graph:
    m,b = np.polyfit(cat_dfs['s'],-cat_dfs['A'],1)
    fit = m*cat_dfs['s']+b
    ax.plot(cat_dfs['s'],fit,'--r')
    mp = m*-100000 #Converting the slope into a % per khrs (*100 to get to %, *1000 to get to khrs,*-1 for degredation)
    ms = f'{round(mp,3)}'
    plt.figtext(0.31,0.26,'Degredation: '+ms+'% /khrs',weight='bold')
    plt.tight_layout()
    plt.show()

def plot_bias_potentio_holds2(area,folder_loc,voltage=True):
    files = os.listdir(folder_loc)
    useful_files = []
    #Making a list of all the files of potentiostatic holds during a bias test
    for file in files:
        if (file.find('PSTAT')!=-1) and (file.find('.DTA')!=-1):
            useful_files.append(file)
    #Getting the first time for reference
    T0_stamp = gt.get_timestamp(os.path.join(folder_loc,'PSTAT_5bias.DTA')) #gets time stamp from first file
    t0 = T0_stamp.strftime("%s") #Conveting Datetime to seconds from Epoch
    #extracting the useful information from the files and placing it into a dataframe
    dfs = [] #Initializing list of dfs
    size = len(useful_files) #gets length of the useful files list
    for i in range(0,size,1):
        loc = os.path.join(folder_loc,useful_files[i]) #Creats a file path to the file of choice
        dta2csv(loc) #convert DTA to a CSV
        loc_csv = loc.replace('.DTA','')+'.csv' #access newly made file
        data = csv.reader(open(loc_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here
        for row in data: #searches first column of each row in csv for "ZCURVE", then adds 1. This gives the right amount of rows to skip
            if row[0] == 'CURVE':
                skip = data.line_num+1
                break
        df = pd.read_csv(loc_csv,sep= '\t',skiprows=skip,encoding='latin1') #creat data frame for a file
        start_time = gt.get_timestamp(loc).strftime("%s") #Find the start time of the file in s from epoch
        df['s'] = df['s'] + int(start_time)
        df_useful = df[['s','A','V vs. Ref.']]
        dfs.append(df_useful)
    cat_dfs = pd.concat(dfs)# (s) Combine all the dataframes in the file folder
    cat_dfs['s'] = (cat_dfs['s']-int(t0))/3600 #(hrs) subtracting the start time to get Delta t and converting time from seconds to hours and
    cat_dfs['A'] = cat_dfs['A'].div(area)
    #plotting:
    if voltage == True:
        # Finding OCV:
        for file in files:
            if (file.find('0bias.DTA')!=-1) and (file.find('OCV')!=-1):
                ocv_path = os.path.join(folder_loc,file)
        dta2csv(ocv_path) #convert DTA to a CSV
        loc_ocv_csv = ocv_path.replace('.DTA','')+'.csv' #access newly made file
        ocv_data = csv.reader(open(loc_ocv_csv, "r",encoding='latin1'), delimiter="\t") #I honestly dk what is going on here        print(ocv_data)
        skip = 0
        for row in ocv_data: #searches first column of each row in csv for "CURVE", then adds 1. This gives the right amount of rows to skip
            if row[0] == 'CURVE':
                skip = ocv_data.line_num+1
                break
        df_ocv = pd.read_csv(loc_ocv_csv,sep= '\t',skiprows=skip,encoding='latin1') #creat data frame for a file
        avg_ocv = df_ocv['V'].mean()
        #Initializing FIgure
        fig,axs = plt.subplots(2)
        #Plotting Bias
        axs[0].set_xlabel('Time (hrs)')
        axs[0].set_ylabel('Voltage (V)')
        axs[0].plot(cat_dfs['s'],cat_dfs['V vs. Ref.'],'.k')
        axs[0].axhline(y=avg_ocv, color= 'r', linestyle='--')
        #Plotting Current Density
        axs[1].set_xlabel('Time (hrs)')
        axs[1].set_ylabel('Current Density (A/cm$^2$)')
        axs[1].plot(cat_dfs['s'],-cat_dfs['A'],'.k')
        #Extras
        axs[1].axhline(y=0, color= 'r', linestyle='--') #Plots 0 Bias on the Current density chart
        plt.figtext(0.15,0.45,'Fuel Cell',weight='bold')
        plt.figtext(0.15,0.35,'Electrolysis',weight='bold')
        plt.tight_layout()
        plt.show()
    else:
        fig,ax = plt.subplots()
        ax.set_xlabel('Time (hrs)')
        ax.set_ylabel('Current Density (A/cm$^2$)')
        ax.plot(cat_dfs['s'],-cat_dfs['A'],'.k')
        #Plot extras
        plt.axhline(y=0, color= 'r', linestyle='--')
        plt.figtext(0.13,0.80,'Fuel Cell Mode',weight='bold')
        plt.figtext(0.13,0.55,'Electrolysis Cell Mode',weight='bold')
        plt.show()

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"DRT functions - streamline the use of Jakes DRT modules"
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def bayes_drt_save(loc_eis,fit_path,fit_name): #takes a potentiostatic EIS spectra and saves it to a certain fit path with a file name
    #Made on 18May21, this is how I currently use Jake's Bayes fit function to fit eis spectra
    #Changes will likely be made in the future as I learn more about DRT and Jakes Modules
    df = gt.read_eis_zdata(loc_eis)
    freq = df['Freq'].values #getting frequency data
    z = df['Zreal'].values + 1j*df['Zimag'].values #getting Zreal and Zimag data
    basis_freq6 = np.logspace(6,-2,81) #getting frequencies to plot over
    inv = Inverter(basis_freq = basis_freq6)
    inv.bayes_fit(freq,z,init_from_ridge=True) #Bayes fitting DRT
    save_pickle(inv,os.path.join(fit_path,fit_name))

def map_drt_save1(loc_eis,fit_path,fit_name): #takes a potentiostatic EIS spectra and saves it to a certain fit path with a file name
    #Made on 18May21, this is how I currently use Jake's Bayes fit function to fit eis spectra
    #Changes will likely be made in the future as I learn more about DRT and Jakes Modules
    df = gt.read_eis_zdata(loc_eis)
    freq = df['Freq'].values #getting frequency data
    z = df['Zreal'].values + 1j*df['Zimag'].values #getting Zreal and Zimag data
    basis_freq6 = np.logspace(6,-2,81) #getting frequencies to plot over
    inv = Inverter(basis_freq = basis_freq6)
    inv.map_fit(freq,z,init_from_ridge=True) #Bayes fitting DRT
    save_pickle(inv,os.path.join(fit_path,fit_name))

def map_drt_save(loc_eis,fit_path,fit_name,which='core'): #takes a potentiostatic EIS spectra and saves it to a certain fit path with a file name
    #Made on 18May21, this is how I currently use Jake's Bayes fit function to fit eis spectra
    #Changes will likely be made in the future as I learn more about DRT and Jakes Modules
    df = gt.read_eis_zdata(loc_eis)
    freq = df['Freq'].values #getting frequency data
    z = df['Zreal'].values + 1j*df['Zimag'].values #getting Zreal and Zimag data
    basis_freq6 = np.logspace(6,-2,81) #getting frequencies to plot over
    inv = Inverter(basis_freq = basis_freq6)
    inv.map_fit(freq,z,init_from_ridge=True) #Bayes fitting DRT
    inv.save_fit_data(os.path.join(fit_path,fit_name),which=which) #main thing that core doesnt save is the matricies (a lot of data)
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"Mass Spec functions"
'Functions to help me format data from the mass spec'
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def ms_df_tc(loc): #Takes a CSV file from the prima DB and turnes it into a df with the useful materials and
    #converts time into relative time from the first measurement
    # Time conversion
    df = pd.read_csv(loc)
    t_init = int(pd.to_datetime(df.at[0,'Time&Date']).strftime("%s")) #converts excel time into something more useable and sets this as the initial time
    df['Time&Date'] = pd.to_datetime(df['Time&Date']).apply(lambda x: x.strftime('%s')) #-t_init#.dt.strftime("%s")-t_init #converts absolute time to relative time
    t_values = [int(t) for t in df['Time&Date'].to_numpy()]
    df['Time&Date'] = [t-t_init for t in t_values]
    return df

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
"Other Functions that dont fit anywhere else"
#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
def lnpo2(ohmic_asr,rp_asr,O2_conc): #Plots ln(1/ASRs) as a function of ln(PO2), imputs are arrays
    # Making ln arrays:
    #pDen = 0.83 #ATM
    ln_O2 = np.log(O2_conc) #*pDen)
    ln_ohmic_asr = np.log(1/ohmic_asr)
    ln_rp_asr = np.log(1/rp_asr)
    # Plotting
    fig,ax = plt.subplots()
    ax.plot(ln_O2,ln_ohmic_asr,'o',color = '#21314D',label = r'ASR$_\mathrm{O}$')
    ax.plot(ln_O2,ln_rp_asr,'o',color = '#D2492A',label = r'ASR$_\mathrm{P}$')
    # Fitting
    mo,bo = np.polyfit(ln_O2,ln_ohmic_asr,1)
    mr,br = np.polyfit(ln_O2,ln_rp_asr,1)
    fit_o = mo*ln_O2 + bo
    fit_r = mr*ln_O2 + br
    ax.plot(ln_O2,fit_o,color = '#21314D')
    ax.plot(ln_O2,fit_r,color = '#D2492A')
    # Formatting
    ax.set_xlabel('ln(O$_2$) (%)')
    ax.set_ylabel('ln(1/ASR) (S/cm$^2$)') #(\u03A9*cm$^2$)
    ax.set_xlim(-1.0,0.1)
    ax.legend()
    #Setting up second x axis
    axx2 = ax.twiny()
    axx2.set_xlabel('Oxygen Concentration (%)')
    axx2.set_xticks(ln_O2)
    axx2.set_xticklabels(O2_conc*100)
    axx2.set_xlim(-1.0,0.1)
    # Figtext
    mo_str = f'{round(mo,2)}'
    plt.figtext(0.5,0.85,r'ASR$_\mathrm{O}$ Slope = '+mo_str,weight='bold')
    mr_str = f'{round(mr,2)}'
    plt.figtext(0.5,0.15,r'ASR$_\mathrm{P}$ Slope = '+mr_str,weight='bold')
    plt.tight_layout()
    plt.show()
