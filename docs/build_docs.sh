#!/bin/bash

sphinx-apidoc -o . ../protocol ../protocol/*/third_party/ ../protocol/*/tests/ 
make singlehtml
