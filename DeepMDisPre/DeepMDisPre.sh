#!/bin/bash
#
SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`
#
## input.fasta
IN="$1"
WDIR=`realpath -s $2`  # working folder
#
ID1=$(basename $IN)
ID=${ID1%.*}

############################################################
# 0.Search msa
############################################################
mkdir -p $WDIR/msa_hhblit
if [ ! -s $WDIR/msa_hhblit/$ID.a3m ]; then
  make_msa.sh $ID.fasta $WDIR/msa_hhblit/$ID $WDIR/msa_hhblit/$ID.a3m 10 10000000
else
  echo "$WDIR/msa_hhblit/$ID.a3m already exist."
fi

############################################################
# 1.Process msa
############################################################
mkdir -p $WDIR/MSA
mkdir -p $WDIR/msa128_random
if [ ! -s $WDIR/MSA/$ID.a3m ]
then
echo "process msa..."
python3.7 $PIPEDIR/process_msa.py $WDIR/msa_hhblit/$ID.a3m $WDIR/MSA/$ID.a3m $WDIR/msa128_random/$ID.a3m
fi

############################################################
# 2.Extract features47
############################################################
mkdir -p $WDIR/Alnstats_feature
mkdir -p $WDIR/Map_feature47

if [ ! -s $WDIR/Alnstats_feature/$ID.colstats ]
then
 echo "extract features47..."
 $PIPEDIR/alnstats $WDIR/MSA/$ID.a3m $WDIR/Alnstats_feature/$ID.colstats $WDIR/Alnstats_feature/$ID.pairstats > /dev/null
 if [ $? != 0 ]; then
   echo "DMP ERROR 09 (alnstats failure) - please report error to psipred@cs.ucl.ac.uk" >&2
   exit 9
 fi

 $PIPEDIR/deepmetapsicov_makepredmap \
 $WDIR/Alnstats_feature/$ID.colstats \
 $WDIR/Alnstats_feature/$ID.pairstats \
 $WDIR/Map_feature47/$ID.deepmetapsicov.map \
 $WDIR/Map_feature47/$ID.deepmetapsicov.fix > /dev/null 2>&1
 if [ $? != 0 ]; then
   echo "DMP ERROR 14 (makepredmap failure) - please report error to psipred@cs.ucl.ac.uk" >&2
   exit 14
 fi
fi

###########################################################
#3. Predict distance
###########################################################
mkdir -p $WDIR/dis_npz

if [ ! -s $WDIR/dis_npz/$ID.npz ]
then
  echo "predict distance..."
  python3.7 $PIPEDIR/predict.py \
  $WDIR/MSA/$ID.a3m \
  $WDIR/Map_feature47/$ID.deepmetapsicov.map \
  $WDIR/msa128_random/$ID.a3m \
  $WDIR/dis_npz/$ID.npz
fi

echo "=======================$ID done============================="