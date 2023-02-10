Private Sub AddBorderToCells()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023

Dim mRange As Range
Dim Cell As Range
Dim v_shift As Long
Dim n_rows As Long

n_rows = ActiveSheet.Rows.Count
n_rows = 3000
v_shift = 4

Set mRange = Range("A" & v_shift, "EK" & n_rows)

Application.EnableEvents = False
Application.ScreenUpdating = False

For Each Cell In mRange
    Cell.BorderAround LineStyle:=xlContinuous, Weight:=xlThin
Next Cell

Application.ScreenUpdating = True
Application.EnableEvents = True

End Sub

Private Sub ColorGrayAnticipated()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023

Dim oRange As Range
Dim mRange As Range
Dim Cell As Range
Dim v_shift As Long
Dim n_rows As Long

Dim col_start As String
Dim col_end As String

Dim condition As Boolean

Dim col_end_int As Long
Dim col_current_int As Long

col_start = "P"

col_end = "EK"
col_end_int = Range(col_end & 1).Column

condition = True


n_rows = ActiveSheet.Rows.Count
n_rows = 3000
v_shift = 4

Set oRange = Range(col_start & v_shift, col_start & n_rows)
Set mRange = oRange

Application.ScreenUpdating = False
Application.EnableEvents = False

While condition

For Each Cell In mRange
    Cell.Interior.ColorIndex = 15
Next Cell

Set mRange = mRange.offset(0, 2)
'col_current = Split(mRange.Address(1, 0), "$")(0)
col_current_int = mRange.Column
condition = col_current_int <= col_end_int

Wend

Application.ScreenUpdating = True
Application.EnableEvents = True


End Sub
Private Sub AddDropDownForVaccines()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023

Dim oRange As Range
Dim mRange As Range
Dim Cell As Range
Dim v_shift As Long
Dim n_rows As Long

Dim col_start As String
Dim col_end As String

Dim condition As Boolean

Dim col_end_int As Long
Dim col_current_int As Long

Dim list_of_vaccines As String
list_of_vaccines = "Pfizer,Moderna,AstraZeneca,COVISHIELD,BPfizerO,BModernaO,Other"

col_start = "AD"
col_end = "AP"

col_end_int = Range(col_end & 1).Column

condition = True


n_rows = ActiveSheet.Rows.Count
n_rows = 3000
v_shift = 4

Set oRange = Range(col_start & v_shift, col_start & n_rows)
Set mRange = oRange

Application.EnableEvents = False
Application.ScreenUpdating = False

While condition

With mRange.Validation
.Delete
.Add Type:=xlValidateList, _
AlertStyle:=xlValidAlertStop, _
Formula1:=list_of_vaccines
End With

Set mRange = mRange.offset(0, 2)
col_current_int = mRange.Column
condition = col_current_int <= col_end_int

Wend

Application.EnableEvents = True
Application.ScreenUpdating = True

End Sub
Private Sub AddDropDownForInfections()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023

Dim oRange As Range
Dim mRange As Range
Dim Cell As Range
Dim v_shift As Long
Dim n_rows As Long

Dim col_start As String
Dim col_end As String

Dim condition As Boolean

Dim col_end_int As Long
Dim col_current_int As Long

Dim list_of_methods As String
list_of_methods = "PCR,RAT"

col_start = "P"
col_end = "AB"

col_end_int = Range(col_end & 1).Column

condition = True


n_rows = ActiveSheet.Rows.Count
n_rows = 3000
v_shift = 4

Set oRange = Range(col_start & v_shift, col_start & n_rows)
Set mRange = oRange

Application.EnableEvents = False
Application.ScreenUpdating = False

While condition

With mRange.Validation
.Delete
.Add Type:=xlValidateList, _
AlertStyle:=xlValidAlertStop, _
Formula1:=list_of_methods
End With

Set mRange = mRange.offset(0, 2)
col_current_int = mRange.Column
condition = col_current_int <= col_end_int

Wend

Application.EnableEvents = True
Application.ScreenUpdating = True

End Sub
Private Sub AddDropDownForStudyRefusals()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023

Dim oRange As Range
Dim mRange As Range
Dim Cell As Range
Dim v_shift As Long
Dim n_rows As Long

Dim col_start As String
Dim col_end As String

Dim condition As Boolean

Dim col_end_int As Long
Dim col_current_int As Long

Dim list_of_methods As String
Dim t_str As String
t_str = "M: Medical history,B: Blood, Q: Questionnaire"
t_str = t_str & ",B & M"
t_str = t_str & ",B & Q"
t_str = t_str & ",M & Q"
t_str = t_str & ",B & M & Q"
list_of_methods = t_str

col_start = "N"
col_end = "N"

col_end_int = Range(col_end & 1).Column

condition = True


n_rows = ActiveSheet.Rows.Count
n_rows = 3000
v_shift = 4

Set oRange = Range(col_start & v_shift, col_start & n_rows)
Set mRange = oRange

Application.EnableEvents = False
Application.ScreenUpdating = False

While condition

With mRange.Validation
.Delete
.Add Type:=xlValidateList, _
AlertStyle:=xlValidAlertStop, _
Formula1:=list_of_methods
End With

Set mRange = mRange.offset(0, 2)
col_current_int = mRange.Column
condition = col_current_int <= col_end_int

Wend

Application.EnableEvents = True
Application.ScreenUpdating = True

End Sub
Private Sub AddDropDownForPhotoConsent()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023

Dim oRange As Range
Dim mRange As Range
Dim Cell As Range
Dim v_shift As Long
Dim n_rows As Long

Dim col_start As String
Dim col_end As String

Dim condition As Boolean

Dim col_end_int As Long
Dim col_current_int As Long

Dim list_of_methods As String
list_of_methods = "Yes,No"

col_start = "L"
col_end = "L"

col_end_int = Range(col_end & 1).Column

condition = True


n_rows = ActiveSheet.Rows.Count
n_rows = 3000
v_shift = 4

Set oRange = Range(col_start & v_shift, col_start & n_rows)
Set mRange = oRange

Application.EnableEvents = False
Application.ScreenUpdating = False

While condition

With mRange.Validation
.Delete
.Add Type:=xlValidateList, _
AlertStyle:=xlValidAlertStop, _
Formula1:=list_of_methods
End With

Set mRange = mRange.offset(0, 2)
col_current_int = mRange.Column
condition = col_current_int <= col_end_int

Wend

Application.EnableEvents = True
Application.ScreenUpdating = True

End Sub
Private Sub AddDropDownForReason()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023

Dim oRange As Range
Dim mRange As Range
Dim Cell As Range
Dim v_shift As Long
Dim n_rows As Long

Dim col_start As String
Dim col_end As String

Dim condition As Boolean

Dim col_end_int As Long
Dim col_current_int As Long

Dim list_of_methods As String
list_of_methods = "Deceased,Discharged,Moved,Declined,Withdrew,Refused-Consent"

col_start = "K"
col_end = "K"

col_end_int = Range(col_end & 1).Column

condition = True


n_rows = ActiveSheet.Rows.Count
n_rows = 3000
v_shift = 4

Set oRange = Range(col_start & v_shift, col_start & n_rows)
Set mRange = oRange

Application.EnableEvents = False
Application.ScreenUpdating = False

While condition

With mRange.Validation
.Delete
.Add Type:=xlValidateList, _
AlertStyle:=xlValidAlertStop, _
Formula1:=list_of_methods
End With

Set mRange = mRange.offset(0, 2)
col_current_int = mRange.Column
condition = col_current_int <= col_end_int

Wend

Application.EnableEvents = True
Application.ScreenUpdating = True

End Sub
Private Sub AddDropDownForSex()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023

Dim oRange As Range
Dim mRange As Range
Dim Cell As Range
Dim v_shift As Long
Dim n_rows As Long

Dim col_start As String
Dim col_end As String

Dim condition As Boolean

Dim col_end_int As Long
Dim col_current_int As Long

Dim list_of_methods As String
list_of_methods = "Female,Male,Other"

col_start = "G"
col_end = "G"

col_end_int = Range(col_end & 1).Column

condition = True


n_rows = ActiveSheet.Rows.Count
n_rows = 3000
v_shift = 4

Set oRange = Range(col_start & v_shift, col_start & n_rows)
Set mRange = oRange

Application.EnableEvents = False
Application.ScreenUpdating = False

While condition

With mRange.Validation
.Delete
.Add Type:=xlValidateList, _
AlertStyle:=xlValidAlertStop, _
Formula1:=list_of_methods
End With

Set mRange = mRange.offset(0, 2)
col_current_int = mRange.Column
condition = col_current_int <= col_end_int

Wend

Application.EnableEvents = True
Application.ScreenUpdating = True

End Sub
Private Sub ForceTextFormat()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023

Range("A4:EK3000").NumberFormat = "@"

End Sub
Private Sub CreateNamedRanges()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 9th 2023

Dim Rng As Range
Dim t_cell As Range
Dim v_shift As Long
Dim n_rows As Long
Dim range_label As String
Dim range_exp As String
Dim t_exp As String
Dim col_start As String
Dim col_end As String
Dim n_events As Long
Dim i As Long



n_rows = ActiveSheet.Rows.Count
n_rows = 3000
v_shift = 4


'==================ID===============
range_label = "ID"
col_start = "C"
col_end = "C"

range_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox range_exp

Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng

'==================NameCol===============
range_label = "NameCol"
col_start = "B"
col_end = "B"

range_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox range_exp

Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng

'==================Sex===============
range_label = "Sex"
col_start = "G"
col_end = "G"

range_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox range_exp

Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng

'==================Site===============
range_label = "Site"
col_start = "A"
col_end = "A"

range_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox range_exp

Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng

'==================Reason===============
range_label = "Reason"
col_start = "K"
col_end = "K"

range_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox range_exp

Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng

'==================Study Refusals===============
range_label = "Refusals"
col_start = "N"
col_end = "N"

range_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox range_exp

Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng

'==================Infection Method===============
range_label = "InfectionMethod"

col_start = "P"
col_end = "P"

range_exp = col_start & v_shift & ":" & col_end & n_rows

'# of infections minus 1
n_events = 6

Set t_cell = Range(col_start & 1)

For i = 1 To n_events
col_start = get_column_name_from_cell(t_cell.offset(0, 2 * i))
col_end = col_start
t_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox t_exp
range_exp = range_exp & "," & t_exp
Next i

'MsgBox range_exp

Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng

'==================Vaccine Type===============
range_label = "VaccineType"

col_start = "AD"
col_end = "AD"

range_exp = col_start & v_shift & ":" & col_end & n_rows

'# of vaccines minus 1
n_events = 6

Set t_cell = Range(col_start & 1)

For i = 1 To n_events
col_start = get_column_name_from_cell(t_cell.offset(0, 2 * i))
col_end = col_start
t_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox t_exp
range_exp = range_exp & "," & t_exp
Next i

'MsgBox range_exp

Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng

'==================Vaccine Date===============
range_label = "VaccineDate"

col_start = "AC"
col_end = "AC"

range_exp = col_start & v_shift & ":" & col_end & n_rows

'# of vaccines minus 1
n_events = 6

Set t_cell = Range(col_start & 1)

For i = 1 To n_events
col_start = get_column_name_from_cell(t_cell.offset(0, 2 * i))
col_end = col_start
t_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox t_exp
range_exp = range_exp & "," & t_exp
Next i

'MsgBox range_exp

Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng


'==================Columns With Dates===============
range_label = "ColumnsWithDates"


'============>DOB
col_start = "H"
col_end = "H"

range_exp = col_start & v_shift & ":" & col_end & n_rows

'============>Study Enrollment Date
col_start = "I"
col_end = "I"

t_exp = col_start & v_shift & ":" & col_end & n_rows
range_exp = range_exp & "," & t_exp

'============>Date Removed from Study
col_start = "J"
col_end = "J"

t_exp = col_start & v_shift & ":" & col_end & n_rows
range_exp = range_exp & "," & t_exp


'============>Questionnaire Completed
col_start = "M"
col_end = "M"

t_exp = col_start & v_shift & ":" & col_end & n_rows
range_exp = range_exp & "," & t_exp

'============>Infections
col_start = "O"
col_end = "O"

t_exp = col_start & v_shift & ":" & col_end & n_rows
range_exp = range_exp & "," & t_exp


'# of infections minus 1
n_events = 6

Set t_cell = Range(col_start & 1)

For i = 1 To n_events
col_start = get_column_name_from_cell(t_cell.offset(0, 2 * i))
col_end = col_start
t_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox t_exp
range_exp = range_exp & "," & t_exp
Next i

'MsgBox range_exp

'============>Vaccination
col_start = "AC"
col_end = "AC"

t_exp = col_start & v_shift & ":" & col_end & n_rows
range_exp = range_exp & "," & t_exp


'# of vaccines minus 1
n_events = 6

Set t_cell = Range(col_start & 1)

For i = 1 To n_events
col_start = get_column_name_from_cell(t_cell.offset(0, 2 * i))
col_end = col_start
t_exp = col_start & v_shift & ":" & col_end & n_rows
'MsgBox t_exp
range_exp = range_exp & "," & t_exp
Next i

'MsgBox range_exp

'============>Blood collection
col_start = "AQ"
col_end = "EK"

t_exp = col_start & v_shift & ":" & col_end & n_rows
range_exp = range_exp & "," & t_exp

'Store range in ColumnsWithDates label
Set Rng = Worksheets("data").Range(range_exp)
ThisWorkbook.Names.Add Name:=range_label, RefersTo:=Rng

End Sub
Function get_column_name_from_cell(Cell As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 8th 2023
'Note that we only want the row to be absolute.
'The vector is zero-based

get_column_name_from_cell = Split(Cell.Address(1, 0), "$")(0)

End Function
Sub hideAllNameTags()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 9th 2023
For Each tempName In ActiveWorkbook.Names
tempName.Visible = False
Next
End Sub
Sub defineLockedCells()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 9th 2023
Cells.Locked = True
Dim v_shift As Long
Dim n_rows As Long
v_shift = 4
n_rows = 3000
Range("A" & v_shift, "EK" & n_rows).Locked = False

End Sub

