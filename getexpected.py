#!/usr/bin/python -B

import re

README_PATH1 = '/tmp/ttt/README'
README_PATH2 = '/tmp/ttt/D0/readme.wav'

EXPECTED_PATH = './expected'
out = open( EXPECTED_PATH, 'w' )

# pattern to match files
pattern = '(\w+?\.(wav|mp3))'
matcher = re.compile( pattern )

def print_both( s1, s2, out ):
    print >> out, ( "MATCH {0} {1}".format( s1, s2 ) )
    print >> out, ( "MATCH {1} {0}".format( s1, s2 ) )

def get_expected_from_file( pathname, out ):
    """Gets the expected matches given a file listing them

    Args:
      pathname: a path name to the README file
      out: a file handle to where this function should write to

    """
    f = open( pathname, 'r' )

    for line in f:
        matches = matcher.findall( line )
        if len(matches) >= 2:
            matches[0] = matches[0][0].strip()
            matches[1] = matches[1][0].strip()
            print_both( matches[0], matches[1], out )

    f.close()

# Exceptions to the pattern that have funny names
excepted_matches = [
        ["z1.ogg", "y1.wav"], \
        ["z2.txt", "y5.wav"], \
        ["z4", "y10.wav"], \
        ["z101", "z01.wav"], \
        ["z111", "z11.mp3"] \
    ]

for matches in excepted_matches:
    print_both( matches[0], matches[1], out )


get_expected_from_file( README_PATH1, out )
get_expected_from_file( README_PATH2, out )

out.close()
