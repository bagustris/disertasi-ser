INCLUDEE=$(shell awk '/^\\include{.*}/ {printf("%s.tex\n", substr($$0,10,length($$0)-10))}' $(BASE).tex)

all: $(BASE).ps

ENV=\
 TEXMFCNF=/usr/local/share/texmf/web2c

MAKEINDEX=mendex

FIG = $(wildcard fig/*.sk)
EPS = $(patsubst %.sk,%.eps,$(FIG))

GREP = /usr/xpg4/bin/egrep
GREPQ = $(GREP) -s

BBL = $(wildcard $(BASE)[.]bbl)

IDX = $(wildcard $(BASE)[.]idx)

ifdef LANDSCAPE
DVIPSFLAGS=-f -t landscape
POSTDVIPS=|pstops '0U(1w,1h)'
XDVIFLAGS=-paper a4r
GVFLAGS=-watch
else
DVIPSFLAGS=-f
POSTDVIPS=
XDVIFLAGS=
GVFLAGS=-watch
endif

dvi: $(BASE).dvi
$(BASE).dvi: $(BASE).tex $(INCLUDEE) $(BBL) $(EPS)

ps8up: $(BASE)-8up.ps
ps4up: $(BASE)-4up.ps
ps2up: $(BASE)-2up.ps
ps: $(BASE).ps
$(BASE).ps: $(BASE).dvi

preview: $(BASE).dvi
	xdvi $(XDVIFLAGS) $< &

preview-gv: $(BASE).ps
	gv $(GVFLAGS) $< &

bib:
	$(BIBTEX) $(BASE)

ind:
	nkf -e $(BASE).idx > /tmp/index$$$$.idx; $(MAKEINDEX) /tmp/index$$$$; cp /tmp/index$$$$.ind $(BASE).ind; rm -f /tmp/index$$$$.idx /tmp/index$$$$.ind /tmp/index$$$$.ilg; cp $(BASE).idx $(BASE).idx.old \

clean:
	rm -f *.aux *.bbl *.blg *.dvi *.toc *.log *.ps *.*pk

.PHONY: all bib ind dvi ps ps2up ps4up ps8up preview preview-gv clean

%-8up.ps: %.ps
	psnup -8 $< $@

%-4up.ps: %.ps
	psnup -4 $< $@

%-2up.ps: %.ps
	psnup -2 $< $@

%.ps: %.dvi
	$(ENV) dvips $(DVIPSFLAGS) $< $(POSTDVIPS) > $@.new && mv $@.new $@

%.eps: %.sk
	sk2ps $*.sk > $*.eps

%.dvi: %.tex
	@echo $(ENV) $(LATEX) $<; \
	[ -f $*.idx ] && ( cmp -s $*.idx $*.idx.old || ( \
	  echo; echo $(MAKEINDEX) $*; \
	  nkf -e $*.idx > /tmp/index$$$$.idx; \
	  $(MAKEINDEX) /tmp/index$$$$; \
	  cp /tmp/index$$$$.ind $*.ind; \
	  rm -f /tmp/index$$$$.idx /tmp/index$$$$.ind /tmp/index$$$$.ilg; cp $*.idx $*.idx.old)); \
	if $(ENV) $(LATEX) $<; then \
	  if $(GREPQ) 'LaTeX Warning: Citation .* undefined' $*.log && $(GREPQ) '\\bibdata' *.aux; then \
	    echo; echo $(BIBTEX) $*; \
	    $(ENV) $(BIBTEX) $*; \
	  fi; \
	  while $(GREP) -v 'LaTeX Font Warning|floatflt Warning' $*.log | $(GREPQ) Warning || \
	        ( cmp -s $*.idx $*.idx.old; [ $$? = 1 ] ); do \
	    tail +2l $*.log > $*.log.bak; \
	    [ -f $*.idx ] && (cmp -s $*.idx $*.idx.old || ( \
	      echo; echo $(MAKEINDEX) $*; \
	      nkf -e $*.idx > /tmp/index$$$$.idx; \
	      $(MAKEINDEX) /tmp/index$$$$; cp /tmp/index$$$$.ind $*.ind; \
	      rm -f /tmp/index$$$$.idx /tmp/index$$$$.ind /tmp/index$$$$.ilg; cp $*.idx $*.idx.old)); \
	    echo; echo try again: $(ENV) $(LATEX) $<; \
	    $(ENV) $(LATEX) $<; \
	    if tail +2l $*.log | cmp -s - $*.log.bak; then \
	      echo; \
	      echo "It is reached to the fixedpoint."; \
	      $(GREPQ) Warning $*.log && (\
	      echo "------------------------------------------------------------"; \
	      $(GREP) Warning $*.log; \
	      echo "------------------------------------------------------------"); \
	      break; \
	    fi; \
	  done; \
	  rm -f $*.log.bak; \
	else \
	  echo; echo "latex exits with an error. $*.dvi is removed."; \
	  rm $*.dvi; \
	  exit 1; \
	fi

check:
	lvgrep '　\|。。\|、、\|がが\|をを\|はは\|未なす\|二ついて' *.tex

