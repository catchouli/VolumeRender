@echo off

set DOCUMENT_NAME="dissertation"

:: CD to batch file directory first
cd %~dp0

:: Create output directories
mkdir int
mkdir bin

:: Run latex, then bibtex, then latex twice again
pdflatex %DOCUMENT_NAME%.tex -output-directory=./bin -aux-directory=./int
biber -U int/%DOCUMENT_NAME%
pdflatex %DOCUMENT_NAME%.tex -output-directory=./bin -aux-directory=./int
pdflatex %DOCUMENT_NAME%.tex -output-directory=./bin -aux-directory=./int