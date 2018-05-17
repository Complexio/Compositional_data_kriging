Sub Main
Debug.Clear

'==================================================
'                User Variables
'==================================================
in_directory = "D:\Onedrive\Documenten\Programming\Python\Jupyter_Notebooks\PROJECTS\M.Sc.Thesis_Matthias\_RESULTS\Test\"
data_file = "Berg_processed_individual.xlsx"
out_directory = "D:\Onedrive\Documenten\Programming\Python\Jupyter_Notebooks\PROJECTS\M.Sc.Thesis_Matthias\_RESULTS\Test\ASCI\"

'==================================================

	Set surf = CreateObject("surfer.application")
	surf.Visible = True

	'Open the data file in Surfer's worksheet and get the last column value
	Dim Wks, WksRange As Object
	Set Wks = surf.Documents.Open(in_directory+data_file)
	Set WksRange = Wks.Rows(1, 1)

	'Loop through z columns, beginning with column 3 and ending at the last column in the data file
	For i=3 To WksRange.ColumnCount
		'Define output grid file directory & name
		grid_file = out_directory + Left(data_file, Len(data_file)-(Len(data_file)-InStrRev(data_file,".")+1) ) + "_zcol"+Str(i)+".asc"

		'Grid the data file
		surf.GridData3(DataFile:= in_directory + data_file, xCol:=1, yCol:=2, zCol:=i, Algorithm:=srfKriging, ShowReport:=False, OutGrid:=grid_file, OutFmt:=srfGridFmtASC)

		Debug.Print "zCol = "+Str(i)
	Next i

End Sub
