Sub Main
	Debug.Clear

'====================
'   User Variables
'====================
file_directory	= "D:\Onedrive\Documenten\Programming\Python\Jupyter_Notebooks\PROJECTS\Compositional_data_kriging\_KRIGING\Kriged_data_classic_aproach\Quarry MHZ\Middle layer"
'====================

	Dim surf, Wks, WksRange, WksRange2, WksRange3 As Object
	Set surf = CreateObject("surfer.application")
	'surf.Visible = True

	'Make sure the file extension has no extra . and the data directory has a trailing \
	file_extension	= LCase(Right(file_extension,(Len(file_extension) - InStrRev(file_extension,"."))))
	If  Len(file_directory)-InStrRev(file_directory,"\") <> 0 Then file_directory = file_directory + "\"

	'Loops throught the GRD files and converts them to DAT
	new_grids = Dir( file_directory  + "*.grd")
	While new_grids <> ""
		'Define output DAT file directory & name
		new_dats = file_directory + Left(new_grids, Len(new_grids)-(Len(new_grids)-InStrRev(new_grids,".")+1) ) + ".asc"

		'Converts the grid file to a DAT file
        surf.GridConvert2(InGrid:= file_directory + new_grids, OutGrid:=new_dats, OutFmt:=srfGridFmtASC)

		Debug.Print new_grids
		new_grids = Dir() 'get next file

	Wend
End Sub
