Sub Main
Debug.Clear

'==================================================
'                User Variables
'==================================================
in_directory = "C:\Users\ENTER PATH\"
data_file = "Lower data.csv"
out_directory = "C:\Users\ENTER PATH\"

'==================================================

	Set SurferApp = CreateObject("Surfer.Application")
	SurferApp.Visible = True

	'Open the data file in Surfer's worksheet and get the last column value
	Dim Wks, WksRange As Object
	Set Wks = SurferApp.Documents.Open(in_directory+data_file)
	Set WksRange = Wks.Rows(1, 1)
	Dim Grid As Object

	'Create a variogram for the interpolation (with a sill of 60 and a range of 0.0015) based on modelled variograms

	Dim VarioComponent As Object
    Set VarioComponent = SurferApp.NewVarioComponent(VarioType:=srfVarGaussian, Param1:=60, Param2:=0.0015)

	'Loop through z columns, beginning with column 12 and ending at the second last column in the data file
	For i=12 To WksRange.ColumnCount-1

		'Define output grid file directory & name (extension left out of the name)
		grid_file = out_directory + Left(data_file, Len(data_file)-(Len(data_file)-InStrRev(data_file,".")+1) ) + "_" + Str(Wks.Cells(1,i))

		'Grid the data files
		SurferApp.GridData2(DataFile:= in_directory + data_file, xCol:=4, yCol:=3, zCol:=i, Algorithm:=srfKriging, ShowReport:=False, KrigVariogram:=VarioComponent, OutGrid:=grid_file + ".grd", OutFmt:=srfGridFmtS7)

		'Create a cross validation report for the data
		SurferApp.CrossValidate2(DataFile:= in_directory + data_file, xCol:=4, yCol:=3, zCol:=i, Algorithm:=srfKriging, Showreport:=True, KrigVariogram:=VarioComponent)

		'Contour map maken van grid file'
		'Declare objects used in script
		Dim Plot, ContourMapFrame, ContourMap As Object

		'Create a plot document in Surfer and assign it to the variable named "Plot"
		Set Plot = SurferApp.Documents.Add(srfDocPlot)

		'Create a countour map and assign the map frame to the "ContourMapFrame" variable
		Set ContourMapFrame = Plot.Shapes.AddContourMap(grid_file)

		'Assign the contour map properties to the variable named "ContourMap"
		Set ContourMap = ContourMapFrame.Overlays(1)

		ContourMap.FillContours = True 'Fill the contour map
		ContourMap.FillForegroundColorMap.LoadFile("C:\Program Files\Golden Software\Surfer 10\ColorScales\Terrain.clr")
		ContourMap.ShowColorScale = True 'Show colorscale
		ContourMap.ApplyFillToLevels(FirstIndex:=1, NumberToSet:=1, NumberToSkip:=0)

		ContourMap.Name = Str(Wks.Cells(1,i))

		'Add locations of drillings to the contour map
		Dim PostLayer, Doc, MapFrame As Object

		Set ContourMapFrame = Plot.Shapes.AddPostMap2(DataFileName:= in_directory + data_file, xCol:= 4, yCol:= 3)
		Set PostLayer = ContourMapFrame.Overlays(1)

		'Plot a title above the contourmap showing the grain size
		Set Shapes = Plot.Shapes
		Dim Text As Object
		Dim Font As Object
		Set Text = Shapes.AddText(x:=4, y:=8.7, Text:=Str(Wks.Cells(1,i)))
		Text.Font.Size = 15 'adjust size of the title


		'If there are negative values, colour the negative contour intervals red
		Dim Levels As Object
		Set Levels = ContourMap.Levels

		For k=1 To 10 '(this value needs to be larger if there are more than ten negative intervals, which is unlikely)
			If Levels(k) < 0 Then
				Levels(k).Fill.ForeColor = srfColorRed
			End If
		Next

		'Print the columnheader for which a grid and contourmap has been made
		Debug.Print "zCol = "+Str(i)

		'Export contourmap as JPEG
		Plot.Export2(FileName:=grid_file + ".jpg" , SelectionOnly:=False , FilterId:="jpg")
	Next i

	'SurferApp.Quit

End Sub


