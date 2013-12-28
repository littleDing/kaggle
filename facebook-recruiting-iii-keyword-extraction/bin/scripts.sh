current=`dirname $0`


function make_tfidf(){
#	python scripts.py --job=load_tf_idf_20131026 --rows=2 --path=../data/Test.csv
#	python scripts.py --job=load_tf_idf_20131026 --rows=1,2 --path=../data/Test.csv
	python scripts.py --job=load_tf_idf_20131026 --rows=1,2 --path=../data/Train.csv
}

function train_model(){
	for loss in log hinge ; do
		for penalty in l1 l2 ; do
			python scripts.py --job=solution_20131102 --loss=$loss --penalty=$penalty ../data/Train.csv
		done
	done
}

function try_output(){
	for loss in log hinge ; do
		for penalty in l1 l2 ; do
			python scripts.py --job=solution_20131102 --loss=$loss --penalty=$penalty --batch=100 ../data/Train.csv 
		done
	done
}

function find_same_title(){
tmp=__tmp__$$__find_same_title
mkdir $tmp
	cat ../data/Train.csv.title_tag ../data/Test.csv.title | sort --parallel=2 -t '	' -T $tmp -k2    \
	| python scripts.py --job=same_title >../data/same_title
rm -rf $tmp
}

function make_cowords(){
tmp=__tmp__$$__make_cowords
mkdir $tmp
	for k in 1 ; do
		python scripts.py --job=make_wid_pairs ../data/Test.csv,../data/Train.csv $k \
		| sort --parallel=2 -n -T $tmp    \
		| python scripts.py --job=count_cowords_reduce >../data/cowords.$k.txt
	done
rm -rf $tmp
}

function make_cotags(){
tmp=__tmp__$$__make_cotags
mkdir $tmp
	for k in 1 2 ; do
		python scripts.py --job=cotags_map $k \
		| sort --parallel=2 -nk1 -T $tmp   \
		| python scripts.py --job=cotags_reduce >../data/cotags.$k.txt
	done
rm -rf $tmp
}

$*
