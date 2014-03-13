ntrain=$1; 
ntest=$2;
tail -n $((ntrain+ntest)) ../temp/svdfeature.input  | head -n $ntrain > ../temp/svdfeature.input.small
tail -n $ntest ../temp/svdfeature.input > ../temp/svdfeature_test.input.small
