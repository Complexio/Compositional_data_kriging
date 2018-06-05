Sub Main
Debug.Clear

'==================================================
'                User Variables
'==================================================
file_directory = "C:\Users\ENTER PATH\"
file_extension = "asc"
out_directory = "C:\Users\ENTER PATH\"

'==================================================

	Set SurferApp = CreateObject("Surfer.Application")
	SurferApp.Visible = True

	'Start the loop
	file_extension	= LCase(Right(file_extension,(Len(file_extension) - InStrRev(file_extension,"."))))
	If  Len(file_directory)-InStrRev(file_directory,"\") <> 0 Then file_directory = file_directory + "\"

	grid_file = Dir( file_directory  + "*." + file_extension)

	On Error GoTo FileError

	While grid_file <> ""
		'Opens new Surfer window
			Dim Plot As Object
			Set Plot = SurferApp.Documents.Add
		'Creates the map
			Dim MapFrame, ContourMap As Object
			Set MapFrame = Plot.Shapes.AddContourMap(GridFileName:=file_directory + grid_file)

			Set ContourMap = MapFrame.Overlays(1)

			ContourMap.FillContours = True 'Fill the contour map
			ContourMap.FillForegroundColorMap.LoadFile("C:\Program Files\Golden Software\Surfer 10\ColorScales\Terrain.clr")
			ContourMap.ShowColorScale = True 'Show colorscale
			ContourMap.ApplyFillToLevels(FirstIndex:=1, NumberToSet:=1, NumberToSkip:=0)

		'Plot a title above the contourmap showing the grain size
			Set Shapes = Plot.Shapes
			Dim Text As Object
			file_name = Left(grid_file, (Len(grid_file) - 4))
			Set Text = Shapes.AddText(x:=3.5, y:=8.7, Text:=file_name)
			Text.Font.Size = 15 'adjust size of the title

		'If there are negative values, colour the negative contour intervals red
			Dim Levels As Object
			Set Levels = ContourMap.Levels

			For k=1 To 10 '(this value needs to be larger if there are more than ten negative intervals, which is unlikely)
				If Levels(k) < 0 Then
					Levels(k).Fill.ForeColor = srfColorRed
				End If
			Next

		'Saves the file as JPEG
			Plot.Export2 (FileName:=out_directory + Left(grid_file, Len(grid_file)-(Len(grid_file)-InStrRev(grid_file,".")+1) )  +".jpg", SelectionOnly:=False, FilterId:="jpg")
		Debug.Print grid_file
			Plot.Close
		grid_file = Dir() 'get next file
	Wend

	SurferApp.Quit
	Exit Sub

	'Error instructions
	FileError: Debug.Print "Error: " + grid_file + " " + Err.Description

	Resume Next

End Sub


