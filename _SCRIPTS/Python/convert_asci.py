import os
import pathlib
import pandas as pd

# CHANGE THESE PARAMETERS TO ALTER THE SCRIPT'S INPUT AND OUTPUT BEHAVIOUR
# ========================================================================
# Set flag as "csv" or "xlsx" to convert .asc files accordingly
flag = "xlsx"
# Set root directory (don't forget to include a forward slash at the end!)
rootDir = "../Python/"
# Set directory to save converted files to (forward slash at end!)
saveDir = "Python/"
# ========================================================================

# Create saveDir if nonexisting
pathlib.Path(saveDir).mkdir(parents=True, exist_ok=True)

# Initializing counters for number of files and folders
n_files = 0
n_folders = 0

# Loop through root directory
for dirName, subdirList, fileList in os.walk(rootDir):
	# Print (sub)directory name
	print(dirName)
	# Increase folder counter by one
	n_folders += 1

	# Loop through files in the active (sub)directory
	for file in fileList:
		# Print filename
		print(file)

		# Check if file's extension is .asc (otherwise do nothing)
		if file.endswith(".asc"):
			# Read in file as csv
			data = pd.read_csv(dirName + "/" + file, 
							  skiprows=6, header=None, sep=" ")
			# Create subdirectory in saveDir according to rootDir tree
			pathlib.Path(saveDir + 
						 dirName[len(rootDir):] + "/").mkdir(parents=True, 
						 									 exist_ok=True)
			# Save as .csv
			if flag == "csv":
				data.to_csv(saveDir + dirName[len(rootDir):] + "/" + file + 
					   	   ".csv", index=False, header=None, sep=";")
			# Save as .xlsx
			if flag == "xlsx":
				writer = pd.ExcelWriter(saveDir + dirName[len(rootDir):] + 
										"/" + file + ".xlsx")
				data.to_excel(writer, index=False, header=False)

			# Increase file counter by one
			n_files += 1


print("\nDone! " + str(n_files) + " file(s) were converted from .asc to ." + 
	  flag + " in " + str(n_folders) + " folder(s)\n")