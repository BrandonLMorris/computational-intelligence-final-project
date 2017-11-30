all: main.tex
	latexmk -pdf main.tex

open: all
	open main.pdf

clean:
	latexmk -C
	rm main.bbl
