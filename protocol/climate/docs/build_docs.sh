#!/bin/bash

sphinx-apidoc -o . .. ../third_party/ ../tests/ 
make singlehtml
