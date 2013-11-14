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



$*
