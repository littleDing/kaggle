FILE=$1

cat $FILE | grep -e "aucs" | sed 's/.*\(aucs=.*\)/\1/g' | head
head $FILE | grep -e "INFO {" | sed 's/.*INFO \(.*\)/\1/g'
cat $FILE | grep -e "train_x=" | tail -n1 | sed 's/.*INFO \(.*\)/\1/g'
cat $FILE | grep -e "INFO 4 fold begins" -e "#117" | awk '
FNR==1{ b=$1" "$2; gsub("-"," ",$1); gsub(":"," ",$2); bt=mktime($1" "$2)}
FNR==2{ e=$1" "$2; gsub("-"," ",$1); gsub(":"," ",$2); et=mktime($1" "$2)}
	END{
	print b,"=>",e,(et-bt)/60
	}'
