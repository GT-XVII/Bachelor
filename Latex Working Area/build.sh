#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Pass 1..."
pdflatex -interaction=nonstopmode --shell-escape dokumentation.tex

echo "Bibliography..."
bibtex dokumentation

echo "Glossaries..."
makeglossaries dokumentation

echo "Pass 2..."
pdflatex -interaction=nonstopmode --shell-escape dokumentation.tex

echo "Pass 3 (final)..."
pdflatex -interaction=nonstopmode --shell-escape dokumentation.tex

echo "Done — dokumentation.pdf"
open dokumentation.pdf
