# A Panoramic Reading of Speculum

This repository holds the scripts used for the analyses reported in the introduction to the supplement on Digital Medieval Studies to [*Speculum. A Journal of Medieval Studies*](http://www.medievalacademy.org/?page=Speculum). This supplement (guest-edited by David Birnbaum, Sheila Bonde and Mike Kestemont) is scheduled to be published in open access in the Fall of 2017. The original data was generously made available to us by the [University of Chicago Press](http://www.journals.uchicago.edu/toc/spc/current) as OCR'ed PDF files or as native XML files. (Note that this data cannot be redistributed here, due to copyright restrictions.) We would like to thank the journal's editor, Sarah Spence for her enthusiast involvement in the project and her valuable feedback. Additionally, we would like to recognize the technical support from Michael Boudreau at the University of Chicago Press.

All scripts under the `src` directory are written for [Python](https://www.python.org/) 3.6 -- we recommend the [Anaconda distribution](https://www.continuum.io/downloads). Any (technical) questions relating to these analyses can be directed to mike [dot] kestemont [at] uantwerp [dot] be. High-quality versions of our final plots can be found in PDF format under the `figures` directory.

Software acknowledgements:
- The text extraction from the OCR'ed PDF files was done using the [Apache PDFBox](https://pdfbox.apache.org/).
- The word clouds were produced using Andreas Mueller's [word cloud package](https://github.com/amueller/word_cloud).
- The texts were tagged using the [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) suite.
- The corpus was wikified using the [Illinois Wikifier](https://cogcomp.cs.illinois.edu/page/software_view/Wikifier).
- Our scripts make extensive use of the excellent [scikit-learn](http://scikit-learn.org/stable/) packages.
