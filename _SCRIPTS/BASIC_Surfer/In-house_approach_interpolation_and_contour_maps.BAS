Sub Main
Debug.Clear

'==================================================
'                User Variables
'==================================================
in_directory = "C:\Users\ENTER PATH\"
data_file = "Lower_transformed.csv"
out_directory = "C:\Users\ENER PATH\"

'==================================================

	Set SurferApp = CreateObject("Surfer.Application")
	SurferApp.Visible = True

	'Open the data file in Surfer's worksheet and get the last column value
	Dim Wks, WksRange As Object
	Set Wks = SurferApp.Documents.Open(in_directory+data_file)
	Set WksRange = Wks.Rows(1, 1)
	Dim Grid As Object

	'Create a variogram for the interpolation (with a sill of 20 and a range of 0.0025) based on the modelled variograms

	Dim VarioComponent As Object
	Set VarioComponent = SurferApp.NewVarioComponent(VarioType:=srfVarSpherical, Param1:=20, Param2:=0.0025)

	'Loop through z columns, beginning with column 4 and ending at the last column in the data file (all 10 PCA values)
	For i=4 To WksRange.ColumnCount
		'Define output grid file directory & name (extension left out of the name)
		grid_file = out_directory + Left(data_file, Len(data_file)-(Len(data_file)-InStrRev(data_file,".")+1) ) + "_" +  Str(Wks.Cells(1,i)) + ".asc"

		'Grid the data files
		SurferApp.GridData2(DataFile:= in_directory + data_file, xCol:=3, yCol:=2, zCol:=i, Algorithm:=srfKriging, ShowReport:=False, KrigVariogram:=VarioComponent, OutGrid:=grid_file, OutFmt:=srfGridFmtASC)

		'Create a cross validation report for the data
		SurferApp.CrossValidate2(DataFile:= in_directory + data_file, xCol:=3, yCol:=2, zCol:=i, Algorithm:=srfKriging, Showreport:=True, KrigVariogram:=VarioComponent)

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

		ContourMap.Name = Str(Wks.Cells(1,i)) + "PCA 10 comp"

		'Plot a title above the contourmap showing the grain size
		Set Shapes = Plot.Shapes
		Dim Text As Object
		Set Text = Shapes.AddText(x:=4, y:=8.5, Text:=Str(Wks.Cells(1,i)))
		Text.Font.Size = 15 'adjust size of the title


		'Print the columnheader for which a grid and contourmap has been made)
		Debug.Print "zCol = "+Str(i)


	Next i

	'SurferApp.Quit

End Sub


