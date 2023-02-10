Private Sub Workbook_SheetChange(ByVal Sh As Object, ByVal Target As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 10th 2023

Dim R As Range

'Exit Sub

'Dim last_row As Long
'Dim last_column As Long
'Dim id_col As Long
'last_column = Cells(1, Columns.Count).End(xlToLeft).Column
' LastColumn = Cells.Find(What:="*",
' After:=Range("A1"), SearchOrder:=xlByColumns, SearchDirection:=xlPrevious).Column
' MsgBox "The last columns is" & last_column

If Not (Sh.Name = "data") Then
Exit Sub
End If

'=============== Site ================
Set R = Intersect(Target, Target.Parent.Range("Site"))
Call validate_site_input(R)

'=============== Name ================
Set R = Intersect(Target, Target.Parent.Range("NameCol"))
Call check_for_empty_id_if_name(R)

'=============== ID ================
Set R = Intersect(Target, Target.Parent.Range("ID"))
Call validate_id_input(R)

'=============== ID to Site ================
Set R = Intersect(Target, Target.Parent.Range("ID"))
Call generate_site_from_id(R)

'=============== Sex ================
Dim list_of_genders() As String
list_of_genders = Split("Female,Male,Other", ",")
Set R = Intersect(Target, Target.Parent.Range("Sex"))
Call validate_list_input(R, list_of_genders, "sex")

'=============== Reason ================
Dim list_of_reasons() As String
list_of_reasons = Split("Deceased,Discharged,Moved,Declined,Withdrew,Refused-Consent", ",")
Set R = Intersect(Target, Target.Parent.Range("Reason"))
Call validate_list_input(R, list_of_reasons, "reason")

'=============== Reason to ABD date =====
'ABD: Anticipated Blood Draw
Set R = Intersect(Target, Target.Parent.Range("Reason"))
Call update_anticipated_blood_draw_from_reason(R)

'=============== Refusals ================
Dim list_of_refusals() As String
Dim t_str As String
t_str = "M: Medical history,B: Blood,Q: Questionnaire"
t_str = t_str & ",B & M"
t_str = t_str & ",B & Q"
t_str = t_str & ",M & Q"
t_str = t_str & ",B & M & Q"
list_of_refusals = Split(t_str, ",")
Set R = Intersect(Target, Target.Parent.Range("Refusals"))
Call validate_list_input(R, list_of_refusals, "refusal")

'=============== Refusals to ABD date =====
'ABD: Anticipated Blood Draw
Set R = Intersect(Target, Target.Parent.Range("Refusals"))
Call update_anticipated_blood_draw_from_refusal(R)

'=============== DOB ================
'=============== DOE ================
'=============== DOR ================
'=============== Vaccine Date ================
'=============== Infection Dates ================
'=============== Blood draw Dates ================
Set R = Intersect(Target, Target.Parent.Range("ColumnsWithDates"))
'Call validate_date_input(R)
Call validate_date_with_month_as_text(R)

'=============== Vaccine Type ================
Dim list_of_vaccines() As String
list_of_vaccines = Split("Pfizer,Moderna,AstraZeneca,COVISHIELD,BPfizerO,BModernaO,Other", ",")
Set R = Intersect(Target, Target.Parent.Range("VaccineType"))
Call validate_list_input(R, list_of_vaccines, "vaccines")

'=============== Infection detection method ================
Dim list_of_methods() As String
list_of_methods = Split("PCR,RAT", ",")
Set R = Intersect(Target, Target.Parent.Range("InfectionMethod"))
Call validate_list_input(R, list_of_methods, "detection method")

'=============== Vaccine Date ================
Set R = Intersect(Target, Target.Parent.Range("VaccineDate"))
Call update_anticipated_collection_date_from_vaccine(R)

End Sub
Private Sub update_anticipated_blood_draw_from_reason(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 10th 2023

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueReasonLoop
End If


If (Not IsEmpty(Cell)) Then
'The participant is no longer active
'We need to update the anticipated blood draw dates

Dim t_string As String
Dim N As Long
Dim i As Long
Dim k As Long
Dim shift As Long
Dim column_to_modify As String

Dim blood_draw_pointers() As String
'Number of doses minus one
N = 7 - 1
ReDim blood_draw_pointers(0 To N)
'Starting point (First blood draw for the first dose)
Set t_cell = Range("AR2")
shift = 14

For i = 0 To N
t_string = get_column_name_from_cell(t_cell.offset(0, shift * i))
blood_draw_pointers(i) = t_string
'MsgBox t_string
Next i

'Dim columns_to_modify() As String
'Columns to modify minus one
N = 7 - 1

'Clean all anticipated dates
For k = 0 To N

column_to_modify = blood_draw_pointers(k)
Set t_cell = Range(column_to_modify & Cell.Row)

For i = 0 To N
column_to_modify = get_column_name_from_cell(t_cell.offset(0, 2 * i))
ActiveSheet.Cells(Cell.Row, column_to_modify).ClearContents
Next i

Next k


End If 'If the Reason cell is not empty

ContinueReasonLoop:
Next Cell ' For each Reason cell

End If ' The region R is not empty

End Sub
Private Sub update_anticipated_blood_draw_from_refusal(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 10th 2023

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueRefusalLoop
End If


If (Not IsEmpty(Cell)) Then
'The participant refused certain procedures
'Did the participant refuse to donate blood?

If Not (Left(Cell.Value, 1) = "B") Then
'B for blood, so the participant DID NOT refuse to donate blood.
GoTo ContinueRefusalLoop
End If

'At this point we know the participant refused to donate blood.

Dim t_string As String
Dim N As Long
Dim i As Long
Dim k As Long
Dim shift As Long
Dim column_to_modify As String

Dim blood_draw_pointers() As String
'Number of doses minus one
N = 7 - 1
ReDim blood_draw_pointers(0 To N)
'Starting point (First blood draw for the first dose)
Set t_cell = Range("AR2")
shift = 14

For i = 0 To N
t_string = get_column_name_from_cell(t_cell.offset(0, shift * i))
blood_draw_pointers(i) = t_string
'MsgBox t_string
Next i

'Dim columns_to_modify() As String
'Columns to modify minus one
N = 7 - 1

'Clean all anticipated dates
For k = 0 To N

column_to_modify = blood_draw_pointers(k)
Set t_cell = Range(column_to_modify & Cell.Row)

For i = 0 To N
column_to_modify = get_column_name_from_cell(t_cell.offset(0, 2 * i))
ActiveSheet.Cells(Cell.Row, column_to_modify).ClearContents
Next i

Next k


End If 'If the Reason cell is not empty

ContinueRefusalLoop:
Next Cell ' For each Reason cell

End If ' The region R is not empty

End Sub

Private Sub generate_site_from_id(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 9th 2023

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueIDLoop
End If

Cell.offset(0, -2).Value = Left(Cell.Value, 2)

ContinueIDLoop:
Next Cell ' For each ID cell

End If ' The region R is not empty

End Sub
Private Sub generate_barcode_from_id(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 9th 2023
'This function is no longer used since
the barcode cannot be directly inferred

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueBarcodeLoop
End If

Cell.offset(0, 1).Value = "LTC1-" & Cell.Value

ContinueBarcodeLoop:
Next Cell ' For each ID cell

End If ' The region R is not empty

End Sub
Private Sub update_anticipated_collection_date_from_vaccine(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 10th 2023
'This function takes as input the vaccination dates as a range.


If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueVDLoop
End If


'Do nothing if the participant is no longer active

Dim t_cell As Range
'Reason column
Set t_cell = Range("K1")

If (Not IsEmpty(Cells(Cell.Row, t_cell.Column))) Then
GoTo ContinueVDLoop
End If


'Do nothing if the participant refused to donate blood

'Study Refusals column
Set t_cell = Range("N1")

If (Not IsEmpty(Cells(Cell.Row, t_cell.Column))) Then

'The refusal column for this row is not empty
If (Left(Cells(Cell.Row, t_cell.Column), 1) = "B") Then
'B for blood, so the participant refused to donate blood.
GoTo ContinueVDLoop
End If

End If



Dim header As String
Dim vac_date_str As String
Dim vac_date As Date
Dim vac_offset As Date
Dim vac_offset_str As String
Dim column_to_modify As String
Dim offset As Long
Dim vac_num As Long
Dim vac_index As Long


Dim t_string As String
Dim N As Long
Dim i As Long
Dim k As Long
Dim shift As Long

vac_date_str = Cell.Value
vac_date = CDate(vac_date_str)
'The headers are located on row #2
header = ActiveSheet.Cells(2, Cell.Column).Value
header = Left(header, 1)
vac_num = CLng(header)
vac_index = vac_num - 1
Dim blood_draw_pointers() As String
'Number of doses minus one
N = 7 - 1
ReDim blood_draw_pointers(0 To N)
'Starting point (First blood draw for the first dose)
Set t_cell = Range("AR2")
shift = 14

For i = 0 To N
t_string = get_column_name_from_cell(t_cell.offset(0, shift * i))
blood_draw_pointers(i) = t_string
'MsgBox t_string
Next i

Dim offset_vec() As Variant
offset_vec = Array(21, 91, 183, 274, 365, 456, 548)

'Dim columns_to_modify() As String
'Columns to modify minus one
N = 7 - 1

'Clean all anticipated dates before the current vaccination
For k = 0 To (vac_index - 1)

column_to_modify = blood_draw_pointers(k)
Set t_cell = Range(column_to_modify & Cell.Row)

For i = 0 To N
column_to_modify = get_column_name_from_cell(t_cell.offset(0, 2 * i))
ActiveSheet.Cells(Cell.Row, column_to_modify).ClearContents
Next i

Next k

'Populate the anticipated dates for the current vaccine
column_to_modify = blood_draw_pointers(vac_index)
Set t_cell = Range(column_to_modify & Cell.Row)

For i = 0 To N

If (IsEmpty(t_cell.offset(0, 2 * i + 1))) Then
'Only compute if the actual blood draw date is unavailable

column_to_modify = get_column_name_from_cell(t_cell.offset(0, 2 * i))
offset = offset_vec(i)
vac_offset = vac_date + offset
vac_offset_str = Format(vac_offset, "dd-mmm-yyyy")
ActiveSheet.Cells(Cell.Row, column_to_modify).Value = vac_offset_str
End If 'Is the actual blood draw date available?

Next i


ContinueVDLoop:
Next Cell ' For each vaccine date cell

End If ' The region R is not empty

End Sub
Function get_column_name_from_cell(Cell As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 3rd 2023
'Note that we only want the row to be absolute.

get_column_name_from_cell = Split(Cell.Address(1, 0), "$")(0)

End Function
Private Sub validate_date_with_month_as_text(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 9th 2023

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueDateLoop
End If

If Not (Len(Cell.Value) = 11) Then
MsgBox "Cell " & Cell.Address & " must have length 11."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

'Day
day_str = Left(Cell.Value, 2)
is_number = IsNumeric(day_str)
If Not (is_number) Then
MsgBox "Cell " & Cell.Address & " must end with 2 digits (day)."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

day_int = CLng(day_str)
If Not (1 <= day_int And day_int <= 31) Then
MsgBox "Cell " & Cell.Address & " day value is unexpected."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

'Dash
dash_character = Mid(Cell.Value, 3, 1)
If Not (dash_character = "-") Then
MsgBox "Cell " & Cell.Address & ": 3rd character should be a dash (-)."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If


'Month
Dim list_of_months() As String
list_of_months = Split("Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec", ",")

Dim found_a_match As Boolean
found_a_match = False

month_str = Mid(Cell.Value, 4, 3)

For Each Item In list_of_months

If (month_str = Item) Then
found_a_match = True
Exit For
End If

Next Item

If Not (found_a_match) Then
MsgBox "Cell " & Cell.Address & " month must have 3 letters and start with uppercase."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If


'Dash
dash_character = Mid(Cell.Value, 7, 1)
If Not (dash_character = "-") Then
MsgBox "Cell " & Cell.Address & ": 7th character should be a dash (-)."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If



'Year
year_str = Right(Cell.Value, 4)
is_number = IsNumeric(year_str)
If Not (is_number) Then
MsgBox "Cell " & Cell.Address & " must start with 4 digits (year)."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

year_int = CLng(year_str)
If Not (1901 <= year_int And year_int <= 2025) Then
MsgBox "Cell " & Cell.Address & " year value is unexpected."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If


ContinueDateLoop:
Next Cell ' For each date (month is text) cell

End If ' The region R is not empty
End Sub
Private Sub validate_date_input(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: January the 30th 2023

'Year-Month-Day
'This function is no longer required

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueDateLoop
End If

If Not (Len(Cell.Value) = 10) Then
MsgBox "Cell " & Cell.Address & " must have length 10."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

'Year
year_str = Left(Cell.Value, 4)
is_number = IsNumeric(year_str)
If Not (is_number) Then
MsgBox "Cell " & Cell.Address & " must start with 4 digits (year)."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

year_int = CLng(year_str)
If Not (1901 <= year_int And year_int <= 2023) Then
MsgBox "Cell " & Cell.Address & " year value is unexpected."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If



'Dash
fifth_character = Mid(Cell.Value, 5, 1)
If Not (fifth_character = "-") Then
MsgBox "Cell " & Cell.Address & ": 5th character should be a dash (-)."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

'Month
month_str = Mid(Cell.Value, 6, 2)
is_number = IsNumeric(month_str)
If Not (is_number) Then
MsgBox "Cell " & Cell.Address & " month must have 2 digits."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

month_int = CLng(month_str)
If Not (1 <= month_int And month_int <= 12) Then
MsgBox "Cell " & Cell.Address & " month value is unexpected."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If


'Dash
eighth_character = Mid(Cell.Value, 8, 1)
If Not (eighth_character = "-") Then
MsgBox "Cell " & Cell.Address & ": 5th character should be a dash (-)."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

'Day
day_str = Right(Cell.Value, 2)
is_number = IsNumeric(day_str)
If Not (is_number) Then
MsgBox "Cell " & Cell.Address & " must end with 2 digits (day)."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

day_int = CLng(day_str)
If Not (1 <= day_int And day_int <= 31) Then
MsgBox "Cell " & Cell.Address & " day value is unexpected."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If


ContinueDateLoop:
Next Cell ' For each date cell

End If ' The region R is not empty
End Sub
Private Sub validate_site_input(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 9th 2023

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueSiteLoop
End If

If Not (Len(Cell.Value) = 2) Then
MsgBox "Cell " & Cell.Address & " must have length 2."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

ContinueSiteLoop:
Next Cell ' For each Site cell

End If ' The region R is not empty

End Sub
Private Sub validate_sex_input(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: January the 30th 2023

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueSexLoop
End If

If Not (Cell.Value = "Male" Or Cell.Value = "Female") Then
MsgBox "Cell " & Cell.Address & " must be Female or Male"
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

ContinueSexLoop:
Next Cell ' For each Sex cell

End If ' The region R is not empty

End Sub
Private Sub validate_id_input(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: January the 30th 2023

If Not R Is Nothing Then

For Each Cell In R

'Imagine we just erased the content of an ID cell.
'If the cell to the left (Name) is not empty,
'make the current cell red.
If (IsEmpty(Cell)) Then

If (Not IsEmpty(Cell.offset(0, -1))) Then
Application.EnableEvents = False
MsgBox "Cell " & Cell.Address & " cannot be empty."
Call unlock_sheet
Cell.Interior.ColorIndex = 3
Call lock_sheet
Application.EnableEvents = True
End If

'Nothging else to do with this cell.
'Move on.
GoTo ContinueIDLoop

End If

If Not (Len(Cell.Value) = 10) Then
MsgBox "Cell " & Cell.Address & " must have length 10."
'vNew = Target.Value
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

first_two_digits = Left(Cell.Value, 2)
is_number = IsNumeric(first_two_digits)
If Not (is_number) Then
MsgBox "Cell " & Cell.Address & " must start with 2 digits."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

last_seven_digits = Right(Cell.Value, 7)
is_number = IsNumeric(last_seven_digits)
If Not (is_number) Then
MsgBox "Cell " & Cell.Address & " must end with 7 digits."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If


third_character = Mid(Cell.Value, 3, 1)
If Not (third_character = "-") Then
MsgBox "Cell " & Cell.Address & ": 3rd character should be a dash (-)."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

'If the cell has a valid ID, then
'color it white.
If Not (IsEmpty(Cell)) Then
Application.EnableEvents = False
Call unlock_sheet
Cell.Interior.ColorIndex = 2
Call lock_sheet
Application.EnableEvents = True
End If

ContinueIDLoop:
Next Cell ' For each ID cell

End If ' The region R is not empty

End Sub
Private Sub validate_reason_input(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: January the 30th 2023

Dim removal_states() As String
Dim found_a_state As Boolean

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueReasonLoop
End If

found_a_state = False

For Each Item In removal_states

If (Cell.Value = Item) Then
found_a_state = True
Exit For
End If

Next Item

If Not (found_a_state) Then
MsgBox "Cell " & Cell.Address & " is not a valid reason."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

ContinueReasonLoop:
Next Cell ' For each Reason cell

End If ' The region R is not empty

End Sub
Private Sub validate_list_input(R As Range, List() As String, obj_type As String)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: January the 30th 2023

Dim found_a_match As Boolean

If Not R Is Nothing Then

For Each Cell In R

If (IsEmpty(Cell)) Then
GoTo ContinueListLoop
End If

found_a_match = False

For Each Item In List

If (Cell.Value = Item) Then
found_a_match = True
Exit For
End If

Next Item

If Not (found_a_match) Then
MsgBox "Cell " & Cell.Address & " is not a valid " & obj_type & " type."
Application.EnableEvents = False
Application.Undo
Application.EnableEvents = True
Exit Sub
End If

ContinueListLoop:
Next Cell ' For each Reason cell

End If ' The region R is not empty

End Sub
Private Sub unlock_sheet()
Dim pword As String
pword = "LTC2021!"
ActiveSheet.Unprotect Password:=pword
End Sub
Private Sub lock_sheet()
Dim pword As String
pword = "LTC2021!"
ActiveSheet.Protect Password:=pword, _
DrawingObjects:=True, _
Contents:=True, Scenarios:=True, _
AllowFormattingCells:=False, AllowFormattingColumns:=True, _
AllowFormattingRows:=False, AllowInsertingColumns:=False, _
AllowInsertingRows:=False, AllowInsertingHyperlinks:=False, _
AllowDeletingColumns:=False, AllowDeletingRows:=False, _
AllowSorting:=False, AllowFiltering:=True, _
AllowUsingPivotTables:=False
End Sub
Private Sub check_for_empty_id_if_name(R As Range)
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 10th 2023

If Not R Is Nothing Then
'Iterate over the name cells
For Each Cell In R
If (Not IsEmpty(Cell)) Then

If (IsEmpty(Cell.offset(0, 1))) Then
'Make the ID cell red if the ID cell is empty
Application.EnableEvents = False
MsgBox "Cell " & Cell.offset(0, 1).Address & " cannot be empty."
Call unlock_sheet
Cell.offset(0, 1).Interior.ColorIndex = 3
Call lock_sheet
Application.EnableEvents = True

End If 'R-Neighbor cell is empty

Else 'The ID cell is not empty
Application.EnableEvents = False
Call unlock_sheet
Cell.offset(0, 1).Interior.ColorIndex = 2
Call lock_sheet
Application.EnableEvents = True


End If 'Name cell is not empty


Next Cell ' For each cell

End If ' The region R is not empty

End Sub
Sub create_report()
'McMaster University
'Analytics Team
'Supervisor: Tara Kajaks
'Developer: Javier Ruiz Ramírez
'Last Revision: February the 10th 2023

Dim ws As Worksheet
Dim report_sheet_exists As Boolean
Dim last_row As Long
Sheets("data").Select

last_row = Cells.Find(What:="*", _
                    After:=Range("A1"), _
                    LookAt:=xlPart, _
                    LookIn:=xlFormulas, _
                    SearchOrder:=xlByRows, _
                    SearchDirection:=xlPrevious, _
                    MatchCase:=False).Row


report_sheet_exists = False

For Each ws In ActiveWorkbook.Worksheets

If (ws.Name = "Report") Then
report_sheet_exists = True
End If

Next ws


If (Not report_sheet_exists) Then
Sheets.Add(After:=Sheets("data")).Name = "Report"
End If

Sheets("Report").Range("A1:A" & last_row).Value = Sheets("data").Range("C1:C" & last_row).Value
Sheets("Report").Range("B1:F" & last_row).Value = Sheets("data").Range("G1:K" & last_row).Value
Sheets("Report").Range("G1:AI" & last_row).Value = Sheets("data").Range("N1:AP" & last_row).Value
Sheets("Report").Select
Rows(1).EntireRow.Delete
Rows(2).EntireRow.Delete
Sheets("Report").Range("A1").Select

End Sub
