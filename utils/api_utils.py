import sys
sys.path.append("..")
import utils.diff_match_patch as dmp_module

def diff_wordMode(text1, text2):
  dmp = dmp_module.diff_match_patch()
  a = dmp.diff_linesToWords(text1, text2)
  lineText1 = a[0]
  lineText2 = a[1]
  lineArray = a[2]
  diffs = dmp.diff_main(lineText1, lineText2, False)
  dmp.diff_charsToLines(diffs, lineArray)
  return diffs