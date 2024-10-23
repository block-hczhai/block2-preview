#! /usr/bin/env bash

if [ "x$1" != "x" ]; then
    RANGE=$1
else
    RANGE=$(seq 0 49)
fi

for i in $RANGE; do
    echo "TEST $(printf "%03d" $i)  " $(head -n 1 $(printf "%03d" $i)-main.in)
    for j in $(grep "#DEP" $(printf "%03d" $i)-main.in | awk '{$1=""; print $0}'); do
        echo -- DEP $(printf "%03d" $j) $(head -n 1 $(printf "%03d" $j)-main.in)
        block2main $(printf "%03d" $j)-main.in > $(printf "%03d" $j)-main.out
        if [ $? -ne 0 ]; then
            cat $(printf "%03d" $j)-main.out
            echo "$(printf "%03d" $j) DEP FAILED!"
            exit 3
        fi
        python3 $(printf "%03d" $j)-check.py $(printf "%03d" $j)-main.out
        rm $(printf "%03d" $j)-main.out
    done
    block2main $(printf "%03d" $i)-main.in > $(printf "%03d" $i)-main.out
    if [ $? -ne 0 ]; then
        cat $(printf "%03d" $i)-main.out
        echo "$(printf "%03d" $i) RUN FAILED!"
        exit 1
    fi
    python3 $(printf "%03d" $i)-check.py $(printf "%03d" $i)-main.out
    if [ $? -ne 0 ]; then
        cat $(printf "%03d" $i)-main.out
        echo "$(printf "%03d" $i) WRONG NUMBER!"
        exit 2
    fi
    rm -r $(printf "%03d" $i)-main.out node0 nodex
done
