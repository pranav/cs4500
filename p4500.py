#!/usr/bin/python

import sys
import wave

def check_args():
  try:
    return len(sys.argv) == 3 and wave.open(sys.argv[1]) and wave.open(sys.argv[2])
  except EOFError:
    print 'ERROR Both files must be of WAV format'
    return True
  except IOError:
    return False


if not check_args():
  print 'ERROR Usage: ./p4500 <path> <path>'


sys.exit(0)

