#!/bin/bash
for FILE in ./plots/*.pdf; do
  pdfcrop ${FILE} ${FILE}
done