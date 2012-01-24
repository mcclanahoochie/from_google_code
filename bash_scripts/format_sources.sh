#!/bin/sh

echo "formatting files..."

ARGS=' --style=java --indent=spaces --pad-oper --pad-header --unpad-paren --add-brackets --add-one-line-brackets --keep-one-line-blocks --keep-one-line-blocks --keep-one-line-statements --align-pointer=type '

find . -name '*.cpp'  -print -exec astyle $ARGS {} \;
find . -name '*.cc'   -print -exec astyle $ARGS {} \;
find . -name '*.c'    -print -exec astyle $ARGS {} \;
find . -name '*.hpp'  -print -exec astyle $ARGS {} \;
find . -name '*.h'    -print -exec astyle $ARGS {} \;
find . -name '*.cu'   -print -exec astyle $ARGS {} \;
find . -name '*.java' -print -exec astyle $ARGS {} \;
find . -name '*.pde'  -print -exec astyle $ARGS {} \;
find . -name '*.cl'   -print -exec astyle $ARGS {} \;

echo "removing astyle crap..."
find . -name '*.orig' -exec rm '{}' \; -print

echo "done"



