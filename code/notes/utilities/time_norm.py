import subprocess
import os
import sys

def get_normalized_time_expressions(anchor, value_list):
    '''Normalizes a list of time expressions with respect to a given anchor'''

    # doesn't guarentee a valid anchor, but should catch most obvious problems
    assert len(anchor) >= 10, "anchor is wrong length %r" % anchor
    # assert len(anchor.split('-')) == 3, "anchor is invalid format %r" % anchor

    # hack to remove time info from achors if present. It is formatted in a way time norm doesn't understand.
    if len(anchor) > 10:
        anchor = anchor[:10]

    values = ""
    for value in value_list:
        values += value + ","

    # remove trailing comma
    values = values[:-1]

    # call subprocess
    output_string = _time_norm(anchor, values)
    output = output_string.split('\n')

    # remove trailing new line
    output = output[:-1]

    return output

def _time_norm(anchor, values):
    '''Calls the timeNorm subprocess using given arguments'''

    timenorm = subprocess.Popen(["scala",
                                "-cp",
                                os.environ["TEA_PATH"] + "/dependencies/TimeNorm/timenorm-0.9.5.jar",
                                os.environ["TEA_PATH"] + "/dependencies/TimeNorm/TimeNorm.scala",
                                anchor,
                                values],
                                cwd=os.environ["TEA_PATH"] + "/dependencies/TimeNorm",
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)

    output, _ = timenorm.communicate()

    return output

if __name__ == "__main__":
    print _time_norm("1995-11-05", "yesterday,tomorrow,today")
    vals = ["yesterday","tomorrow", "today"]
    print get_normalized_time_expressions("1995-11-05", vals)
