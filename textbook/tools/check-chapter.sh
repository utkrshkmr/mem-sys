#!/usr/bin/env bash
# Compile a single chapter on its own to catch LaTeX errors quickly.
# Usage: tools/check-chapter.sh chapters/05-delta-rule-family.tex
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
base=$(basename "$1" .tex)
num=${base%%-*}
out="$here/build/check-$base"
rm -rf "$out"; mkdir -p "$out"
bibs="$here/bib/core"
[ -f "$here/bib/c$num.bib" ] && bibs="$bibs,$here/bib/c$num"
cat > "$out/check.tex" <<TEX
\\documentclass[11pt,twoside,openright]{book}
\\input{$here/preamble}
\\begin{document}
\\mainmatter
\\input{$here/chapters/$base}
\\bibliographystyle{plainnat}
\\bibliography{$bibs}
\\end{document}
TEX
cd "$out"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error check.tex > latexmk.out 2>&1
status=$?
echo "== $base: latexmk exit $status (log: $out/check.log)"
grep -E '^(/|\./).*:[0-9]+: |^! ' check.log | head -20
grep -A3 '^! ' check.log | head -20
echo "-- undefined citations (must fix):"
grep -oE "Citation \`[^']+' (on page [0-9]+ )?undefined" check.log | sort -u | head -30
echo "-- undefined references (fine if they point to other chapters):"
grep -oE "Reference \`[^']+' on page [0-9]+ undefined" check.log | sed -E 's/ on page [0-9]+//' | sort -u | head -30
grep -E '^(Package|LaTeX) .*Error' check.log | head
[ -f check.pdf ] && echo "-- pages: $(pdfinfo check.pdf 2>/dev/null | awk '/^Pages/{print $2}')"
exit $status
