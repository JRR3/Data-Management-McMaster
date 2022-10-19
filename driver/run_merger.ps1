$dt = [datetime]::now.tostring("yyyy_MM_dd_T_H_m_s")
$fname = "py_log_$dt.txt"
python .\merge_driver.py > $fname
