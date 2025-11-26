# FM24 Position Rating Calculator

Personal hobby tool for **Football Manager 2024** that converts player attributes into
FIFA-style position ratings.

The workflow is:

1. Use relevant custom FM view file (`.fmf`) so you can see all the player attributes in said screen.
2. In-game, select a group of players, press **Ctrl+A**, then **Ctrl+P** to "Print" the view to a text file.
3. Drag the exported file onto `Run-PosCalc.cmd`.
4. The script calls `PosCalc.py`, which parses the attributes and outputs:
   - A `.csv` file with raw ratings
   - An `.html` report in the `reports/` folder

This is mainly for a fun test and to speed up squad planning.
In theory should be able to be used on any Football Manager game where the attributes are the same and custom views are available.
Weightings of attributes per position can be reordered on the .py file source code.

---

## Files in this repo

- `PosCalc.py`  
  Main Python script. Reads printed attribute files, converts attributes into ratings for multiple positions
  using custom weightings, and writes the results to CSV and HTML.

- `Run-PosCalc.cmd`  
  Windows helper script so you can drag-and-drop one or more exported files onto it.
  It will:

  - Use `venv\Scripts\python.exe` if a virtual environment is present.
  - Otherwise fall back to the system `python`.
  - Run `PosCalc.py` once for each dropped file.

- `lfc26attributes.rtf`  
  Example attribute export file created from FM24.  
  You can use this as a demo input even if you don’t own the game:
  
  ```text
  1. Clone this repo
  2. Set up Python (see below)
  3. Drag lfc26attributes.rtf onto Run-PosCalc.cmd
  4. Open the generated HTML report in the reports/ folder
